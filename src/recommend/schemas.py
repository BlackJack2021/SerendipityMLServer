from pydantic import BaseModel
from typing import List


class RecommendedByIMF(BaseModel):
    edinet_codes: List[str]
