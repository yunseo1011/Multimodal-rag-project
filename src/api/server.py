#src/api/server.py
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.core.model_loader import get_model
from src.api.routers import embedding, health

# Lifespan Event: 서버 시작/종료 시 실행될 작업
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server Starting... Loading Model...")
    # 서버 켤 때 모델을 미리 로드해둠 (Warm-up)
    get_model()
    yield
    print("Server Shutting down...")

app = FastAPI(
    title="Multimodal RAG Embedding API",
    description="LayoutLMv3 기반 문서 임베딩 서버",
    version="1.0.0",
    lifespan=lifespan
)

# 라우터 등록
app.include_router(embedding.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn
    # 로컬 개발용 실행
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)