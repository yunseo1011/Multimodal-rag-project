# ocr_service / api.py

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

from ocr_service.aggregator import OCRAggregator

# 1. 모델 로딩
# 전역 변수로 선언하여 요청 때마다 모델을 다시 로드하지 않도록 함
ocr_aggregator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global ocr_aggregator
    print("OCR 모델 로딩 중...")
    ocr_aggregator = OCRAggregator()
    print("OCR 모델 로딩 완료!")
    
    # 임시 폴더 생성
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    yield
    
    # 종료 시 실행 (필요하다면 리소스 정리)
    print("서버 종료")

app = FastAPI(title="OCR Service API", lifespan=lifespan)

# 2. API 엔드포인트 정의
@app.post("/ocr", summary="이미지 OCR 수행", description="이미지 파일을 업로드하여 텍스트 추출 결과를 반환합니다.")
def extract_text(file: UploadFile = File(...)):
    """
    [동기 함수 사용 이유]
    OCR은 CPU-bound 작업입니다. 
    async def가 아닌 def로 선언하면 FastAPI가 자동으로 ThreadPool에서 실행하여
    이벤트 루프가 차단되는 것을 방지합니다.
    """
    
    # 1. 파일 확장자 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 2. 임시 파일 저장 (OCRAggregator가 파일 경로를 요구하므로)
    # 파일명 충돌 방지를 위해 UUID 사용
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    temp_file_path = os.path.join("temp", temp_filename)

    try:
        # 업로드된 파일 내용을 임시 파일에 씀
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. OCR 실행
        # (전역 변수로 로드된 모델 사용)
        result = ocr_aggregator.run(temp_file_path)
        
        # 4. 결과 반환
        return {
            "status": "success",
            "filename": file.filename,
            "document": result.dict() # Pydantic 모델을 dict로 변환
        }

    except Exception as e:
        # 에러 발생 시 로그 출력 및 500 에러 반환
        print(f"OCR 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
        
    finally:
        # 5. 뒷정리: 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "OCR Service is running"}