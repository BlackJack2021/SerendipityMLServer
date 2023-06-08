from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.recommend.ml.imf.production_model.config import parameter
from src.recommend.ml.imf.predictor import predict_for_user as imf_predictor
from src.recommend.ml.item2vec.predictor import predict_for_user as item2vec_predictor
from src.recommend.ml.association_rule.predictor import (
    predict_for_user as association_rule_predictor,
)
import src.recommend.schemas as schemas
from src.db import get_db

router = APIRouter()


@router.get(
    "/ml/recommend/imf/predict/{user_id}", response_model=schemas.RecommendedByIMF
)
def predict_by_imf(user_id: str):
    """Implicit Matrix Factorization モデルによるレコメンドを実施"""
    factors = parameter["factors"]
    n_epochs = parameter["n_epochs"]
    k = parameter["k"]
    recommended = imf_predictor(
        user_id=user_id, factors=factors, n_epochs=n_epochs, k=k
    )
    return {"edinet_codes": recommended}


@router.get(
    "/ml/recommend/item2vec/predict/{user_id}",
    response_model=schemas.RecommendedByItem2Vec,
)
async def predict_by_item2vec(user_id: str, db: AsyncSession = Depends(get_db)):
    recommended = await item2vec_predictor(user_id=user_id, db=db)
    return {"edinet_codes": recommended}


@router.get(
    "/ml/recommend/association_rule/predict/{user_id}",
    response_model=schemas.RecommendedByAssociationRule,
)
async def predict_by_association_rule(user_id: str, db: AsyncSession = Depends(get_db)):
    recommended = await association_rule_predictor(user_id=user_id, db=db)
    return {"edinet_codes": recommended}
