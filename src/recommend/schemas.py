from pydantic import BaseModel
from typing import List


class RecommendedByIMF(BaseModel):
    edinet_codes: List[str]


class RecommendedByItem2Vec(BaseModel):
    edinet_codes: List[str]


class RecommendedByAssociationRule(BaseModel):
    edinet_codes: List[str]
