import sys
import os
import json
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 모듈 임포트 
from src.core.model_loader import get_model
from src.core.ocr_processor import process_ocr_data  # 순수 데이터 파싱
from src.core.embedding import extract_embedding     # 텐서 변환 + 추론 + Mean Pooling

# 경로 설정
DATA_DIR = os.path.join(project_root, "data/raw")
OCR_DIR = os.path.join(project_root, "data/processed/ocr")
OUTPUT_DIR = os.path.join(project_root, "data/processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_full_text(json_path):
    """
    JSON에서 텍스트 추출 (full_text 키 사용)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, dict) and "full_text" in data:
                return data["full_text"]
                
        return "" # full_text 없으면 빈 문자열

    except Exception as e:
        return ""

def main():
    print("Ingest Pipeline 시작...")

    # 1. 모델 로딩
    model, processor = get_model()

    # 2. 이미지 파일 찾기
    image_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"   -> 총 {len(image_files)}개 파일 처리 예정")
    data_list = []

    # 3. 루프 시작
    for image_path in tqdm(image_files, desc="Processing"):
        try:
            # 파일 경로 매칭
            rel_path = os.path.relpath(image_path, DATA_DIR)
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # JSON 경로
            json_path = os.path.join(OCR_DIR, f"{file_name}.json")
            
            # JSON 없으면 혹시 하위 폴더에 있나 확인
            if not os.path.exists(json_path):
                json_path = os.path.join(OCR_DIR, os.path.dirname(rel_path), f"{file_name}.json")
                if not os.path.exists(json_path):
                    continue

            # 로직 수행       
            # 1. Raw Data 로드 (process_ocr_data)
            ocr_result = process_ocr_data(image_path, json_path)
            
            # 2. 임베딩 추출 (extract_embedding: Processor + Mean Pooling 포함)
            embedding = extract_embedding(model, processor, ocr_result)
            
            # 3. 텍스트 로드 
            full_text = load_full_text(json_path)

            # 데이터 추가
            data_list.append({
                "doc_id": f"{file_name}_{len(data_list)}",
                "file_path": image_path,
                "embedding": embedding,
                "text": full_text,  # DB에 원본 텍스트 저장
                "metadata": {"json_path": json_path}
            })

        except Exception as e:
            print(f"❌ Error ({file_name}): {e}")
            continue

    # 4. 저장
    if data_list:
        df = pd.DataFrame(data_list)
        save_path = os.path.join(OUTPUT_DIR, "document_embeddings.parquet")
        df.to_parquet(save_path, index=False)
        print(f"✅ 저장 완료: {save_path} ({len(df)}건)")
    else:
        print(" 처리된 데이터가 없습니다.")

if __name__ == "__main__":
    main()