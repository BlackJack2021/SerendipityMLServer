import pandas as pd
from typing import List
from collections import Counter
import pickle

from sqlalchemy.ext.asyncio import AsyncSession
from src.recommend.ml.association_rule.fetcher import fetch_data_for_predict


async def predict_for_user(
    db: AsyncSession,
    user_id: str,
    k: int = 10,
    model_path: str = "./src/recommend/ml/association_rule/production_model/association_rule.pkl",
) -> List[str]:
    raw_user_df = await fetch_data_for_predict(db, user_id)
    with open(model_path, "rb") as f:
        association_rule: pd.DataFrame = pickle.load(f)
    user_evaluated_items = raw_user_df["edinet_code"].unique().tolist()
    latest_evaluated_items = raw_user_df.sort_values("datetime", ascending=False)[
        "edinet_code"
    ].tolist()

    # 新しい閲覧企業に対し、順次レコメンドを実装していく
    recommends: List[str] = []
    for item in latest_evaluated_items:
        # 当該企業が条件部に一本でも含まれているアソシエーションルールを抽出
        matched_flags = association_rule["antecedents"].apply(
            lambda x: len(set([item]) & set(x)) >= 1
        )
        # アソシエーションルールの帰結部をアイテムリストに格納
        consequent_items = []
        for i, row in (
            association_rule[matched_flags]
            .sort_values("lift", ascending=False)
            .iterrows()
        ):
            print(row)
            consequent_items.extend(row["consequents"])
        counter = Counter(consequent_items)
        for edinet_code, edinet_code_count in counter.most_common():
            if edinet_code not in user_evaluated_items:
                recommends.append(edinet_code)
            if len(recommends) == k:
                break
        if len(recommends) == k:
            break
    return recommends
