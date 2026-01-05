# scripts/ingest.py
import sys
import os
import json
import warnings
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# 경고 무시
warnings.filterwarnings("ignore")

load_dotenv()

# 프로젝트 루트 경로 설정 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

from src.core.embedding import get_embedding 

# 경로 설정
OCR_DIR = os.path.join(project_root, "data/processed/ocr")
OUTPUT_DIR = os.path.join(project_root, "data/processed")
# 원본 이미지 경로 (나중에 출처 보여줄 때 필요)
RAW_DIR = os.path.join(project_root, "data/raw") 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_full_text(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "full_text" in data:
                return data["full_text"]
            if "lines" in data:
                return " ".join([line.get("text", "") for line in data["lines"]])
            return ""
    except Exception as e:
        print(f" 텍스트 로드 실패: {json_path} / {e}")
        return ""

def main():
    print(" Gemini 기반 임베딩 생성 시작...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: .env 파일에 GOOGLE_API_KEY가 없습니다.")
        return

    data_list = []
    
    # OCR 폴더 탐색
    json_files = []
    for root, dirs, files in os.walk(OCR_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"   -> 총 {len(json_files)}개 문서 처리 예정")

    for json_path in tqdm(json_files, desc="Processing"):
        try:
            file_name = os.path.splitext(os.path.basename(json_path))[0]
            label = os.path.basename(os.path.dirname(json_path))

            text_content = load_full_text(json_path)
            if len(text_content) < 5:
                continue

            embedding = get_embedding(text_content)

            image_path = os.path.join(RAW_DIR, label, file_name + ".png")
            
            data_list.append({
                "doc_id": file_name,
                "text": text_content,
                "embedding": embedding,
                "label": label,
                "file_path": image_path, 
                "metadata": {
                    "json_path": json_path
                }
            })

        except Exception as e:
            print(f"❌ Error ({file_name}): {e}")
            continue

    if data_list:
        df = pd.DataFrame(data_list)
        save_path = os.path.join(OUTPUT_DIR, "document_embeddings.parquet")
        
        print(f"✅ 저장 완료: {save_path}")
        print(f"   - 총 문서 수: {len(df)}") # 994개로 6개 걸러짐
        df.to_parquet(save_path, index=False)
    else:
        print(" 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()