from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import gensim

from src.recommend.ml.utils.dataclass import Dataset, RecommendResult
from src.recommend.ml.utils.metric_calcurator import MetricCalculator
from src.recommend.ml.item2vec.schemas import raw_data_for_train_schema


class Trainer:
    """Word2Vec をユーザーの行動履歴情報に適用し、各企業の性質の分散表現を取得"""

    def __init__(
        self,
        minimum_num_per_user: Optional[int] = 3,
        valid_ratio: Optional[float] = 1 / 3,
        factors: int = 100,
        n_epochs: int = 30,
        window: int = 2,
        use_skip_gram: int = 1,
        use_hierarchial_softmax: int = 0,
        min_count: int = 1,
        k: int = 10,
        end_date: Optional[datetime] = datetime(2023, 5, 31),
    ):
        """
        モデルのハイパーパラメータなど、学習及び検証に必要なパラメータを渡す
        Argument:
            minimum_num_per_user: Optional[int] = 3
                1ユーザー当たり最低何本の企業閲覧が存在することを前提とするか
                プロダクション用のモデルを構築する際にはここを None にする
            valid_ratio: Optional[float] = 1/3
                検証データの比率
                プロダクション用のモデルを構築する際にはここを None にする
            factors: int = 100
                単語ベクトルの次元数
            n_epochs: int = 30
                学習における重みパラメータの更新回数
            window: int = 2
                ある単語の周辺語を何単語までとするか
            use_skip_gram: int = 1
                ニューラルネットワークモデルとして SkipGram を使用するかどうか
            use_hierarchial_softmax: int = 0
                階層的ソフトマックス法を使用するかどうか。 use_skip_gram=1 ならこちらは0にすべき。
            min_count: int = 1
                企業数の出現回数の閾値。今回はせいぜい5000事業者ぐらいなので
                初期値ではフィルタリングは敢えて行わないものとする。
            k: int = 10
                Precision@K や Recall@K における K
            end_date: Optional[datetime] = datetime(2023, 5, 31)
                データをそろえるため、使用するデータの最新日付を指定する
        """
        self.minimum_num_per_user = minimum_num_per_user
        self.valid_ratio = valid_ratio
        self.factors = factors
        self.n_epochs = n_epochs
        self.window = window
        self.use_skip_gram = use_skip_gram
        self.use_hierarchial_softmax = use_hierarchial_softmax
        self.min_count = min_count
        self.k = k
        self.end_date = end_date

        # モデルを格納する
        self.model: Optional[gensim.models.word2vec.Word2Vec] = None

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

    def fit(
        self,
        train_df: pd.DataFrame,
        factors: int,
        n_epochs: int,
        window: int,
        use_skip_gram: int,
        use_hierarchial_softmax: int,
        min_count: int,
    ):
        """学習データに基づいてモデルの学習を実施"""
        # まずは train_df から各ユーザーの閲覧企業のリストを作成
        item2vec_data: List[str] = []
        for user_id, user_df in train_df.groupby("user_id"):
            item2vec_data.append(
                user_df.sort_values("datetime")["edinet_code"].tolist()
            )

        # モデルの学習を実施
        model = gensim.models.word2vec.Word2Vec(
            item2vec_data,
            vector_size=factors,
            window=window,
            sg=use_skip_gram,
            hs=use_hierarchial_softmax,
            epochs=n_epochs,
            min_count=min_count,
            workers=1,  # word2vec の出力を determinisitic にするために必要
        )
        self.model = model

    def predict(self, train_df: pd.DataFrame, k: int) -> RecommendResult:
        """学習したモデルに基づいて k 個商品をレコメンドする"""
        pred_user2items = dict()
        for user_id, user_df in train_df.groupby("user_id"):
            input_data = []
            for item_id in user_df.sort_values("datetime")["edinet_code"].tolist():
                if item_id in self.model.wv.key_to_index:
                    input_data.append(item_id)
                if len(input_data) == 0:
                    pred_user2items[user_id] = []
                    continue
            recommended_items = self.model.wv.most_similar(input_data, topn=k)
            pred_user2items[user_id] = [d[0] for d in recommended_items]
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
        self.fit(
            train_df=dataset.train_df,
            factors=self.factors,
            n_epochs=self.n_epochs,
            window=self.window,
            use_skip_gram=self.use_skip_gram,
            use_hierarchial_softmax=self.use_hierarchial_softmax,
            min_count=self.min_count,
        )
        recommend_result = self.predict(train_df=dataset.train_df, k=self.k)
        metrics = self.evaluate(
            valid_user2items=dataset.valid_user2items,
            pred_user2items=recommend_result.pred_user2items,
            k=self.k,
        )
        return metrics

    def train_and_save_model_for_production(
        self,
        raw_df: pd.DataFrame,
        model_path: str = "./src/recommend/ml/item2vec/production_model/item2vec.model",
    ):
        """プロダクション用のモデルの学習及び保存を実施"""
        df = raw_data_for_train_schema.validate(raw_df)
        # 利用できる全てのデータを用いて学習を実施
        self.fit(
            train_df=df,
            factors=self.factors,
            n_epochs=self.n_epochs,
            window=self.window,
            use_skip_gram=self.use_skip_gram,
            use_hierarchial_softmax=self.use_hierarchial_softmax,
            min_count=self.min_count,
        )
        # matrix_info と model の保存を実施
        self.model.save(model_path)

        print("モデルの学習・保存が完了しました。")
