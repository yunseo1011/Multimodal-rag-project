#src/api/routers/embedding.py
import os
import time
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

# Core 모듈 가져오기
from src.core.model_loader import get_model
from src.core.ocr_processor import process_ocr_data
from src.core.embedding import extract_embedding

router = APIRouter(prefix="/api/v1", tags=["Embeddings"])

# 임시 파일 저장소
TEMP_DIR = "temp_api"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/embeddings", summary="문서 임베딩 추출")
async def create_embedding(
    image: UploadFile = File(...),
    ocr_json: UploadFile = File(...)
):
    """
    [입력] 이미지 파일 + OCR 결과 JSON 파일
    [출력] 768차원 임베딩 벡터
    """
    start_time = time.time()
    
    # 1. 고유 ID 생성 (파일명 충돌 방지)
    request_id = str(uuid.uuid4())
    img_ext = image.filename.split(".")[-1]
    
    temp_img_path = os.path.join(TEMP_DIR, f"{request_id}.{img_ext}")
    temp_json_path = os.path.join(TEMP_DIR, f"{request_id}.json")

    try:
        # 2. 임시 파일 저장
        with open(temp_img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        with open(temp_json_path, "wb") as f:
            shutil.copyfileobj(ocr_json.file, f)

        # 3. 모델 로드 (Singleton이라 빠름)
        model, processor = get_model()

        # 4. 데이터 전처리 (Core 모듈 사용)
        ocr_data = process_ocr_data(temp_img_path, temp_json_path)

        # 5. 임베딩 추출 (Core 모듈 사용)
        vector = extract_embedding(model, processor, ocr_data)

        if vector is None:
            raise HTTPException(status_code=500, detail="Vector extraction failed")

        # 6. 결과 반환
        return {
            "filename": image.filename,
            "vector": vector,
            "dimension": len(vector),
            "process_time": round(time.time() - start_time, 4)
        }

    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 7. 뒷정리 (임시 파일 삭제)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)