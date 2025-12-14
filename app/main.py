from fastapi import FastAPI

#1. FastAPI 앱 인스턴스 생성
app = FastAPI(
    title = "Multimodal RAG server"
)

#2. 기본경로(Root)
@app.get("/")
async def health_check():
    return {"status":"OK", "message":"Server is up and running!"}