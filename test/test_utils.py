from utils import get_data_pairs

RAW_ROOT = "data/raw"
OCR_ROOT = "data/processed/ocr"

# 1. 매핑 실행
pairs = get_data_pairs(RAW_ROOT, OCR_ROOT)

# 2. 결과 샘플 출력
if len(pairs) > 0:
    print("\n--- 첫 번째 데이터 샘플 ---")
    print(pairs[0])
    
    # 3. 라벨 분포 확인
    from collections import Counter
    labels = [p['label'] for p in pairs]
    print("\n--- 클래스별 데이터 개수 ---")
    print(Counter(labels))
else:
    print("데이터를 찾을 수 없습니다. 경로를 확인해주세요.")