import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm # 진행률 표시바 


# 설정 (Configuration)
TARGET_TOTAL = 1000  # 총 다운로드 목표 개수
OUTPUT_DIR = "data/raw" # 이미지 저장 경로
METADATA_FILE = "data/metadata.json" # 메타데이터 파일 경로


#  메인 로직
def main():
    # 1. 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"폴더 생성 완료: {OUTPUT_DIR}")

    # 2. 데이터셋 스트리밍 연결 
    print("HuggingFace 연결 중...")
    dataset = load_dataset("rvl_cdip", split="train", streaming=True, trust_remote_code=True)
    
    # 클래스 정보 가져오기 (0: letter, 1: form ...)
    labels = dataset.features['label'].names
    num_classes = len(labels)
    target_per_class = (TARGET_TOTAL // num_classes) + 1 # 1000장을 채우기 위해 여유분 확보 (+1)
    
    print(f"목표: 총 {TARGET_TOTAL}장 (클래스당 약 {target_per_class}장)")

    # 3. 다운로드 루프
    counters = {name: 0 for name in labels} # 클래스별 개수 세는 카운터
    metadata = [] # 메타데이터 저장할 리스트
    total_saved = 0
    
    # 진행률 표시바 (tqdm)
    pbar = tqdm(total=TARGET_TOTAL, desc="Downloading")

    try:
        for i, sample in enumerate(dataset):
            # 목표 달성하면 종료
            if total_saved >= TARGET_TOTAL:
                break

            label_id = sample['label']
            label_name = labels[label_id]

            # 해당 클래스가 이미 목표치를 채웠으면 건너뜀 (균형 맞추기)
            if counters[label_name] >= target_per_class:
                continue

            try:
                # --- ** 이미지 처리 및 검증 ** ---
                image = sample['image']
                
                # 깨진 이미지 검사 (verify)
                image.verify() 
                
                # 흑백 -> RGB 변환
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # --- 저장 경로 설정 ---
                # data/raw/invoice/ 폴더가 없으면 생성
                class_dir = os.path.join(OUTPUT_DIR, label_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                
                # 파일명: doc_0001.png
                file_name = f"doc_{total_saved:04d}.png"
                save_path = os.path.join(class_dir, file_name)
                
                # 이미지 저장
                image.save(save_path)

                # --- 메타데이터 기록 ---
                meta = {
                    "id": f"doc_{total_saved:04d}",
                    "image_path": save_path,
                    "label_id": label_id,
                    "label_name": label_name
                }
                metadata.append(meta)

                # 카운터 업데이트
                counters[label_name] += 1
                total_saved += 1
                pbar.update(1)

            except Exception as e:
                print(f"⚠️ 에러 발생 (Skipping): {e}")
                continue
    
    finally:
        pbar.close()

        # 4. 메타데이터 JSON 파일 저장
        print(f" 메타데이터 저장 중... ({METADATA_FILE})")
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n 다운로드 및 메타데이터 생성이 끝났습니다!")
        print(f"총 저장된 이미지: {total_saved}장")

if __name__ == "__main__":
    main()