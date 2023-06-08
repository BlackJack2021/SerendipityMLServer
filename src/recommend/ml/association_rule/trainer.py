import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict, Counter
import pickle

from src.recommend.ml.imf.schemas import raw_data_for_train_schema
from src.recommend.ml.utils.dataclass import Dataset, RecommendResult
from src.recommend.ml.utils.metric_calcurator import MetricCalculator


class Trainer:
    def __init__(
        self,
        minimum_num_per_user: Optional[int] = 5,
        valid_ratio: Optional[float] = 1 / 5,
        min_support: float = 0.011,
        metric: str = "lift",
        min_threshold: float = 1.0,
        k: int = 10,
        end_date: Optional[datetime] = datetime(2023, 5, 31),
    ):
        """
        Argument:
            minimum_num_per_user: Optional[int] = 10
                1ユーザー当たり最低何本の企業閲覧が存在することを前提とするか
                プロダクション用のモデルを構築する際にはここを None にする
            valid_ratio: Optional[float] = 1/3
                検証データの比率
                プロダクション用のモデルを構築する際にはここを None にする
            min_support: float = 0.011
                支持度の最小値を指定
            metric: str = 'lift'
                アソシエーションルールの並び替え指標を決定
            k: int = 10
                Precision@K や Recall@K における K
            end_date: datetime = datetime(2023, 5, 31)
                利用するデータの最終日情報
        """
        self.minimum_num_per_user = minimum_num_per_user
        self.valid_ratio = valid_ratio
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.k = k
        self.end_date = end_date
        self.model: Optional[pd.DataFrame] = None

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
            user_df = user_df.drop_duplicates(subset=["edinet_code"])
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
        min_support: float,
        metric: str,
        min_threshold: float,
    ):
        """学習データを使ってアソシエーションルールを計算する"""
        # まずはユーザー×アイテムのマトリックスを作成する
        user_item_matrix = train_df.drop_duplicates(["user_id", "edinet_code"]).pivot(
            index="user_id", columns="edinet_code", values="filer_name"
        )
        user_item_matrix[user_item_matrix.isnull()] = False
        user_item_matrix[user_item_matrix != False] = True
        for col in user_item_matrix.columns.tolist():
            user_item_matrix[col] = user_item_matrix[col].astype(bool)
        # 支持度で絞り込みを行い、アソシエーションルールを計算
        print(user_item_matrix.shape)
        freq_items = apriori(
            user_item_matrix, min_support=min_support, use_colnames=True
        )
        print(freq_items.shape)
        rules = association_rules(
            freq_items, metric=metric, min_threshold=min_threshold
        )
        self.model = rules

    def predict(self, train_df: pd.DataFrame, k: int) -> RecommendResult:
        user_evaluated_items = (
            train_df.groupby("user_id")
            .agg({"edinet_code": list})["edinet_code"]
            .to_dict()
        )
        pred_user2items = defaultdict(list)
        for user_id, user_df in train_df.groupby("user_id"):
            # ユーザーが直近評価した5つの企業を取得
            latest_evaluated_items = user_df.sort_values("datetime")[
                "edinet_code"
            ].tolist()[-5:]
            # それらの企業が条件部に1本でも含まれているアソシエーションルールを抽出
            matched_flags = self.model["antecedents"].apply(
                lambda x: len(set(latest_evaluated_items) & set(x)) >= 1
            )
            # アソシエーションルールの帰結部をアイテムリストに格納。
            # その後登場頻度順に並び替え、まだユーザーが評価していないものがあればそれを追加
            consequent_items = []
            for i, row in (
                self.model[matched_flags]
                .sort_values("lift", ascending=False)
                .iterrows()
            ):
                consequent_items.extend(row["consequents"])
            counter = Counter(consequent_items)
            for edinet_code, edinet_code_count in counter.most_common():
                if edinet_code not in user_evaluated_items[user_id]:
                    pred_user2items[user_id].append(edinet_code)
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
        df = raw_data_for_train_schema.validate(raw_df)
        dataset = self.preprocess(
            raw_df=df,
            minimum_num_per_user=self.minimum_num_per_user,
            valid_ratio=self.valid_ratio,
            end_date=self.end_date,
        )
        self.fit(
            train_df=dataset.train_df,
            min_support=self.min_support,
            metric=self.metric,
            min_threshold=self.min_threshold,
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
        model_path: str = "./src/recommend/ml/association_rule/production_model/association_rule.pkl",
    ):
        """プロダクション用のモデルの学習及び保存を実施"""
        train_dfs = []
        df = raw_data_for_train_schema.validate(raw_df)
        for user_id, user_df in df.groupby("user_id"):
            user_df = user_df.drop_duplicates(["edinet_code"])
            if len(user_df) < self.minimum_num_per_user:
                continue
            train_dfs.append(user_df)
        train_df = pd.concat(train_dfs)
        self.fit(
            train_df=train_df,
            min_support=self.min_support,
            metric=self.metric,
            min_threshold=self.min_threshold,
        )
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
