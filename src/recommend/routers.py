from fastapi import APIRouter
from typing import List

from src.recommend.ml.imf.production_model.config import parameter
from src.recommend.ml.imf.predictor import predict_for_user
import src.recommend.schemas as schemas

router = APIRouter()


@router.get(
    "/ml/recommend/imf/predict/{user_id}", response_model=schemas.RecommendedByIMF
)
def predict_by_imf(user_id: str):
    """Implicit Matrix Factorization モデルによるレコメンドを実施"""
    factors = parameter["factors"]
    n_epochs = parameter["n_epochs"]
    k = parameter["k"]
    recommended = predict_for_user(
        user_id=user_id, factors=factors, n_epochs=n_epochs, k=k
    )
    return {"edinet_codes": recommended}
