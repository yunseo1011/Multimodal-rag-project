import os
from ocr_service.aggregator import OCRAggregator

# 경로 설정 (최상위 루트 지정)
INPUT_ROOT = "data/raw"            # 이 아래에 invoice, form 등 하위 폴더가 있다고 가정
OUTPUT_ROOT = "data/processed/ocr" # 결과를 저장할 루트

def main():
    print(" 전체 데이터 OCR 일괄 처리 시작...")
    
    # 1. 모델 로딩
    aggregator = OCRAggregator()
    
    total_processed = 0

    # 2. os.walk로 모든 하위 폴더 순회
    # os.walk는 폴더 트리를 재귀적으로 탐색합니다.
    for root, dirs, files in os.walk(INPUT_ROOT):
        
        # 이미지 파일만 골라내기
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        print(f"\n 현재 폴더 처리 중: {root}")
        
        for filename in image_files:
            img_path = os.path.join(root, filename)
            
            # 1. 현재 파일이 raw 폴더 기준 어디에 있는지 계산 (예: "invoice")
            relative_path = os.path.relpath(root, INPUT_ROOT)
            
            # 2. 저장할 폴더 경로 생성 (예: "data/processed/ocr/invoice")
            target_dir = os.path.join(OUTPUT_ROOT, relative_path)
            
          
            try:
                # OCR 실행 (신뢰도 0.6, 줄간격 15)
                result = aggregator.run(
                    img_path, 
                    confidence_threshold=0.6,
                    row_tolerance=15
                )
                
                # 해당 폴더에 저장
                aggregator.save_to_json(result, target_dir)
                
                print(f"[성공] {filename} -> {target_dir}/")
                total_processed += 1
                
            except Exception as e:
                print(f"[실패] {filename}: {e}")

    print(f"\n전체 처리 완료! 총 {total_processed}개의 문서를 변환했습니다.")

if __name__ == "__main__":
    main()