import numpy as np
from typing import Dict, List
from src.recommend.ml.utils.dataclass import Metrics


class MetricCalculator:
    def _precision_at_k(
        self, true_items: List[int], pred_items: List[int], k: int
    ) -> float:
        """あるユーザーのPrecision@K を計算

        Precision@K とは、「k個の商品おすすめのうち、ユーザーが高評価するアイテムの割合がどれだけだったか」を表す指標。
        """
        if k == 0:
            return 0.0
        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    def _recall_at_k(
        self, true_items: List[int], pred_items: List[int], k: int
    ) -> float:
        """あるユーザーの Recall@K を計算

        Recall@K とは、「ユーザーが高評価するアイテム群のうち、どれくらいの割合を k 個の商品おすすめで網羅できたか」を表す指標。
        """
        if len(true_items) == 0 or k == 0:
            return 0.0
        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
        return r_at_k

    def _calc_recall_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        """各ユーザーの Recall@K を計算し、平均値を計算"""
        scores = []
        # 各ユーザーの recall@K を計算
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(r_at_k)
        return np.mean(scores)

    def _calc_precision_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        """各ユーザーの Precision@K を計算し、平均値を計算"""
        scores = []
        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(p_at_k)
        return np.mean(scores)

    def calc(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(precision_at_k, recall_at_k)
