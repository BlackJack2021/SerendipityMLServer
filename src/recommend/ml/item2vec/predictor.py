import gensim
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from src.recommend.ml.item2vec.fetcher import fetch_data_for_predict


async def predict_for_user(
    db: AsyncSession,
    user_id: str,
    k: int = 10,
    model_path: str = "./src/recommend/ml/item2vec/production_model/item2vec.model",
) -> List[str]:
    raw_user_df = await fetch_data_for_predict(db, user_id)
    model = gensim.models.word2vec.Word2Vec.load(model_path)
    input_data = []
    for item_id in raw_user_df.sort_values("datetime")["edinet_code"].tolist():
        if item_id in model.wv.key_to_index:
            input_data.append(item_id)
        if len(input_data) == 0:
            return []
    recommended_items = model.wv.most_similar(input_data, topn=k)
    recommended = [d[0] for d in recommended_items]
    return recommended
