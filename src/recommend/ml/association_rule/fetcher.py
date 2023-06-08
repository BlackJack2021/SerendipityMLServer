from sqlalchemy import text
from src.db import sync_session
import pandas as pd
from src.recommend.ml.imf.schemas import raw_data_for_train_schema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Result


def fetch_data_for_train() -> pd.DataFrame:
    """データベースから学習に必要なデータを取得"""
    with open("./src/recommend/ml/imf/sqls/fetch_data_for_train.sql", "r") as f:
        sql = f.read()

    with sync_session() as session:
        data = session.execute(text(sql))

    raw_df = pd.DataFrame(data.all())
    checked_df = raw_data_for_train_schema.validate(raw_df)
    return checked_df


async def fetch_data_for_predict(db: AsyncSession, user_id: str) -> pd.DataFrame:
    with open("./src/recommend/ml/item2vec/sqls/fetch_data_for_predict.sql", "r") as f:
        query = f.read()
    result: Result = await db.execute(text(query), {"user_id": user_id})
    raw_df = pd.DataFrame(result.all())
    return raw_df
