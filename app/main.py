from fastapi import FastAPI
from app.data_loader import get_random_document

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title = "Multimodal RAG server"
)

# 기본경로(Root)
@app.get("/")
async def health_check():
    return {"status":"OK", "message":"Server is up and running!"}

@app.get("/random-doc")
def get_sample():
    image, label = get_random_document()

    return {
        "status": "success",
        "label": label,
        "image_size": image.size,
        "message": "이미지를 성공적으로 로드했습니다! (나중에 AI에게 넘겨질 예정)"
    }
