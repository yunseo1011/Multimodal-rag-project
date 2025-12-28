#scr/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional
'''API가 주고받을 데이터의 명세서'''

# 응답(Response) 모델
class EmbeddingResponse(BaseModel):
    filename: str
    vector: List[float]    # 768차원 벡터
    dimension: int         # 벡터 길이 (768)
    process_time: float    # 처리 시간 (선택 사항)

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool