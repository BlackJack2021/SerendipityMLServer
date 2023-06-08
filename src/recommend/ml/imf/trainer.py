import pandas as pd
from typing import List, Optional, Dict
import implicit
from scipy.sparse import lil_matrix
from collections import defaultdict
from datetime import datetime
import pickle

from src.recommend.ml.imf.schemas import raw_data_for_train_schema
from src.recommend.ml.utils.dataclass import Dataset, MatrixInfo, RecommendResult
from src.recommend.ml.utils.metric_calcurator import MetricCalculator


class Trainer:
    def __init__(
        self,
        minimum_num_per_user: Optional[int] = 3,
        valid_ratio: Optional[float] = 1 / 3,
        alpha: float = 1.0,
        factors: int = 30,
        n_epochs: int = 50,
        random_state: int = 1,
        regularization: float = 0.1,
        k: int = 10,
        end_date: Optional[datetime] = datetime(2023, 5, 31),
    ):
        """
        モデルの学習に用いるクラスを提供

        Argument:
            minimum_num_per_user: Optional[int] = 3
                1ユーザー当たり最低何本の企業閲覧が存在することを前提とするか
                プロダクション用のモデルを構築する際にはここを None にする
            valid_ratio: Optional[float] = 1/3
                検証データの比率
                プロダクション用のモデルを構築する際にはここを None にする
            alpha: float = 1.0
                信頼度を算出するのに用いる比例定数. 詳しくは description.md を参照すること
            factors: int = 10
                行列因子分解に用いる因子数
            n_epochs: int = 50
                学習におけるパラメータの更新回数
            random_state: int = 1
                implicit.alt.AlternatingLeastSquare の乱数シードの設定
            k: int = 10
                Precision@K や Recall@K における K
            regularization: float = 0.1
                正則化項の重みパラメータ
            end_date: datetime = datetime(2023, 5, 31)
                利用するデータの最終日情報
        """
        self.minimum_num_per_user = minimum_num_per_user
        self.valid_ratio = valid_ratio
        self.alpha = alpha
        self.factors = factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.k = k
        self.regularization = regularization
        self.end_date = end_date
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None

    @staticmethod
    def preprocess(
        raw_df: pd.DataFrame,
        minimum_num_per_user: Optional[int],
        valid_ratio: Optional[float],
        end_date: Optional[datetime],
    ):
        """前処理パート

        生データを引数とし、以下を実行する
        1. 学習データと検証データに分割する
        2. 学習データと検証データのそれぞれで、閲覧企業のリストの辞書を作成する
        """
        # end_date が指定されている場合、その日付より前のデータのみを解析対象とする
        if end_date is not None:
            df = raw_df[raw_df["datetime"] <= end_date]
        else:
            df = raw_df.copy()
        # 学習データに minimum 個以上閲覧履歴が存在するユーザーを対象とする
        # 各学習データを時間順に並び替えて、新しいものから valid_ratio を検証データとする。
        trains: List[pd.DataFrame] = []
        valids: List[pd.DataFrame] = []
        for _, user_df in df.groupby("user_id"):
            if len(user_df) < minimum_num_per_user:
                continue
            valid_num = int(len(user_df) * valid_ratio)
            uid_valid_df = user_df.sort_values(by="datetime", ascending=False).iloc[
                :valid_num
            ]
            uid_train_df = user_df.sort_values(by="datetime", ascending=False).iloc[
                valid_num:
            ]
            valids.append(uid_valid_df)
            trains.append(uid_train_df)
        train_df = pd.concat(trains)
        valid_df = pd.concat(valids)
        valid_user2items = (
            valid_df.groupby("user_id")
            .agg({"edinet_code": list})["edinet_code"]
            .to_dict()
        )

        return Dataset(
            train_df=train_df, valid_df=valid_df, valid_user2items=valid_user2items
        )

    @staticmethod
    def get_matrix_info(
        train_df: pd.DataFrame,
        alpha: float,
    ):
        """学習データを user × company の行列形式で持つ際の諸情報を取得"""
        unique_user_ids = sorted(train_df["user_id"].unique())
        unique_item_ids = sorted(train_df["edinet_code"].unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        item_id2index = dict(zip(unique_item_ids, range(len(unique_item_ids))))
        matrix = lil_matrix((len(unique_item_ids), len(unique_user_ids)))
        for _, row in train_df.iterrows():
            user_id = row["user_id"]
            item_id = row["edinet_code"]
            user_index = user_id2index[user_id]
            item_index = item_id2index[item_id]
            matrix[item_index, user_index] = 1.0 * alpha
        return MatrixInfo(
            user_id2index=user_id2index, item_id2index=item_id2index, matrix=matrix
        )

    def fit(
        self,
        matrix: lil_matrix,
        factors: int,
        n_epochs: int,
        random_state: int,
        regularization: float,
    ):
        """モデルの学習を実施"""
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=n_epochs,
            calculate_training_loss=True,
            random_state=random_state,
            regularization=regularization,
        )
        self.model.fit(matrix.T.tocsr())

    def predict(self, matrix_info: MatrixInfo, k: int) -> RecommendResult:
        """学習したモデルに基づいて k 個商品をレコメンドする"""
        recommendations = self.model.recommend_all(matrix_info.matrix.T.tocsr())
        pred_user2items = defaultdict(list)
        unique_item_ids = list(matrix_info.item_id2index.keys())
        for user_id, user_index in matrix_info.user_id2index.items():
            item_indexes = recommendations[user_index, :]
            # 推薦個数が k 個になるまで推薦を続ける
            for item_index in item_indexes:
                item_id = unique_item_ids[item_index]
                pred_user2items[user_id].append(item_id)
                if len(pred_user2items[user_id]) == k:
                    break
        return RecommendResult(pred_user2items)

    def evaluate(
        self,
        valid_user2items: Dict[str, List[str]],
        pred_user2items: Dict[str, List[str]],
        k: int,
    ):
        """metric_calculator に基づいて精度指標を計算"""
        metric_calculator = MetricCalculator()
        metrics = metric_calculator.calc(
            true_user2items=valid_user2items, pred_user2items=pred_user2items, k=k
        )
        print(metrics)
        return metrics

    def execute_experiment(self, raw_df: pd.DataFrame):
        """raw_df を入力とし、学習から評価まで実施"""
        df = raw_data_for_train_schema.validate(raw_df)
        dataset = self.preprocess(
            raw_df=df,
            minimum_num_per_user=self.minimum_num_per_user,
            valid_ratio=self.valid_ratio,
            end_date=self.end_date,
        )
        matrix_info = self.get_matrix_info(train_df=dataset.train_df, alpha=self.alpha)
        self.fit(
            matrix=matrix_info.matrix,
            factors=self.factors,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            regularization=self.regularization,
        )
        recommend_result = self.predict(matrix_info=matrix_info, k=self.k)
        metrics = self.evaluate(
            valid_user2items=dataset.valid_user2items,
            pred_user2items=recommend_result.pred_user2items,
            k=self.k,
        )
        return metrics

    def train_and_save_model_for_production(
        self,
        raw_df: pd.DataFrame,
        matrix_info_path: str = "./src/recommend/ml/imf/production_model/matrix_info.pkl",
        model_path: str = "./src/recommend/ml/imf/production_model/imf_model",
    ):
        """プロダクション用のモデルの学習及び保存を実施"""
        df = raw_data_for_train_schema.validate(raw_df)
        matrix_info = self.get_matrix_info(train_df=df, alpha=self.alpha)
        self.fit(
            matrix=matrix_info.matrix,
            factors=self.factors,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            regularization=self.regularization,
        )
        # matrix_info と model の保存を実施
        with open(matrix_info_path, "wb") as f:
            pickle.dump(matrix_info, f)
        self.model.save(model_path)

        print("モデルの学習・保存が完了しました。")
