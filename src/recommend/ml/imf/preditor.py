import implicit
import pickle
from typing import List

from src.recommend.ml.utils.dataclass import MatrixInfo


def predict_for_user(
    user_id: str,
    factors: int,
    n_epochs: int,
    k: int,
    matrix_info_path: str = "./src/recommend/ml/imf/production_model/matrix_info.pkl",
    model_path: str = "./src/recommend/ml/imf/production_model/imf_model.pkz",
) -> List[str]:
    # 行列情報のロード
    with open(matrix_info_path, "rb") as f:
        matrix_info: MatrixInfo = pickle.load(f)

    # 学習データにユーザーがいるかどうかを確認し、存在していない場合は何も返さない
    if user_id not in matrix_info.user_id2index():
        return []

    # モデルのロードを行う
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=n_epochs,
        calculate_training_loss=True,
        random_state=1,
    ).load(model_path)
    # レコメンドするアイテムのリストを返す
    recommend_item_indexes = model.recommend(userid=user_id, N=k)
    unique_item_ids = list(matrix_info.item_id2index.keys())
    recommend_ids = []
    for recommend_item_index in recommend_item_indexes:
        recommend_ids.append(unique_item_ids[recommend_item_index])
    return recommend_ids
