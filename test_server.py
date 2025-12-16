import os
import json
import random
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

# 설정
METADATA_FILE = "data/metadata.json"

# 1. 메타데이터 로드
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"데이터 로드 완료: {len(dataset)}개")
else:
    dataset = []
    print("메타데이터 파일이 없습니다.")

@app.get("/")
def read_root():
    return {"message": "멀티모달 RAG 데이터 서버가 정상 작동 중입니다"}

@app.get("/random")
def get_random_image():
    """새로고침 할 때마다 랜덤 이미지를 보여줍니다."""
    if not dataset:
        return {"error": "데이터가 없습니다."}
    
    # 랜덤 뽑기
    item = random.choice(dataset)
    image_path = item['image_path']
    label = item['label_name']
    
    # 이미지 파일이 실제로 있는지 확인 후 전송
    if os.path.exists(image_path):
        print(f"전송 중: {label} - {image_path}") # 터미널 로그용
        return FileResponse(image_path)
    else:
        return {"error": "파일을 찾을 수 없습니다.", "path": image_path}

# 실행은 터미널에서: uvicorn test_server:app --reload