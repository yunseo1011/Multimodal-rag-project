from pydantic import BaseModel, Field 
from typing import List, Optional

# 임베딩 결과
class EmbeddingResponse(BaseModel):
    filename: str
    vector: List[float]    # 768차원 벡터
    dimension: int         # 벡터 길이 (768)
    process_time: float    # 처리 시간 (선택 사항)

# 서버 상태 확인
class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

# 검색 요청 (Request)
class SearchRequest(BaseModel):
    query: str = Field(..., example="invoice from 2023", description="검색할 질문")
    top_k: int = Field(5, example=5, description="가져올 문서 개수")
    filter_label: Optional[str] = Field(None, example="invoice", description="특정 라벨(invoice, resume 등)만 필터링")

# 검색 결과 아이템 (개별 문서)
class SearchResultItem(BaseModel):
    rank: int
    doc_id: str
    score: float
    label: str
    file_path: str
    text: str

# 최종 검색 답변 (Response)
class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# 채팅
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"  # <-- 사용자 구별용 ID (기본값 설정)

class ChatResponse(BaseModel):
    response: str
    category: str
    reason: str