import dataclasses
import pandas as pd
from typing import Dict, List
from scipy.sparse import lil_matrix

@dataclasses.dataclass(frozen=True)
class Dataset:
    # 学習用の評価値データセット
    train_df: pd.DataFrame
    # テスト用の評価値データセット
    valid_df: pd.DataFrame
    # key はユーザーID, value = 実際に閲覧した企業リスト
    valid_user2items: Dict[str, List[str]]

@dataclasses.dataclass(frozen=True)
class MatrixInfo:
    user_id2index: Dict[str, int]
    item_id2index: Dict[str, int]
    matrix: lil_matrix

@dataclasses.dataclass(frozen=True)
# 推薦システムの予測結果
class RecommendResult:
    # key は user_id, value はおすすめアイテムIDのリスト。Precision@K や Recall@K の計算に利用
    pred_user2items: Dict[int, List[int]]

@dataclasses.dataclass(frozen=True)
# 推薦システムの各評価指標の値を格納
class Metrics:
    precision_at_k: float
    recall_at_k: float
    # 評価結果を出力するときに少数派第3桁までにする
    def __repr__(self):
        return f"Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"