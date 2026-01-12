import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from api.routers.search import router as search_router  
from api.routers.chat import router as chat_router

# 앱 생성
app = FastAPI(
    title="Multimodal RAG API",
    description="이미지와 텍스트를 동시에 이해하는 검색 엔진 API",
    version="1.0"
)

# 라우터 등록 
# prefix="/api/v1" -> 실제 주소는 localhost:8000/api/v1/search
app.include_router(search_router, prefix="/api/v1", tags=["Search"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)