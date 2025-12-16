import json
import os
from PIL import Image
from tqdm import tqdm

METADATA_FILE = "data/metadata.json"

def main():
    print("이미지 무결성 검사(Corrupt Check) 시작...")
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrupt_files = []
    
    for item in tqdm(data):
        try:
            with Image.open(item['image_path']) as img:
                img.verify() # 내부 데이터 손상 확인
        except Exception as e:
            corrupt_files.append(item['image_path'])
            print(f"⚠️ 손상된 파일 발견: {item['image_path']} ({e})")

    print("\n" + "="*30)
    if not corrupt_files:
        print("완벽합니다! 깨진 이미지(Corrupt Image)가 0장입니다.")
    else:
        print(f"총 {len(corrupt_files)}장의 깨진 파일이 발견되었습니다.")
        # 필요시 삭제 로직 추가 가능
    print("="*30)

if __name__ == "__main__":
    main()