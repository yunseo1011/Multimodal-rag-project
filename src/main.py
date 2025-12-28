import sys
import os

# 1. 프로젝트 루트 경로 설정 (어디서 실행하든 동작하게 함)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 2. 모듈 불러오기
from src.core.model_loader import get_model
from src.core.ocr_processor import process_ocr_data
from src.core.embedding import extract_embedding
from src.utils.data_utils import get_data_pairs 

def main():
    print("System initializing...")
    
    # 1. 모델 로딩
    model, processor = get_model()
    
    # 2. 싱글톤 확인
    print("Checking Singleton...")
    model2, _ = get_model()
    if model is model2:
        print("✅ Singleton Verified!")
    
    # 3. 실제 데이터 자동 탐색 
    print("\n Searching for test data...")
    raw_root = os.path.join(project_root, "data/raw")
    ocr_root = os.path.join(project_root, "data/processed/ocr")
    
    # 데이터 폴더에서 짝꿍(이미지+JSON)이 맞는 거 아무거나 가져옴
    pairs = get_data_pairs(raw_root, ocr_root)
    
    if len(pairs) > 0:
        # 첫 번째 데이터로 테스트
        sample = pairs[0]
        print(f"Testing with: {os.path.basename(sample['image_path'])}")
        
        try:
            # 전처리
            ocr_data = process_ocr_data(sample['image_path'], sample['json_path'])
            
            # 임베딩 추출
            vector = extract_embedding(model, processor, ocr_data)
            
            print(f"\n✅ Vector Extraction Successful!")
            print(f"   - Dimensions: {len(vector)}")
            print(f"   - Sample Values: {vector[:5]} ...")

            
        except Exception as e:
            print(f"❌ Processing Error: {e}")
    else:
        print("⚠️ data 폴더에 매칭되는 파일(이미지+JSON)이 하나도 없습니다.")
        print("   data/raw 와 data/processed/ocr 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()