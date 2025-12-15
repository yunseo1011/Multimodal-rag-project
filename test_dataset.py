# test_dataset.py
from datasets import load_dataset
import io

def inspect_rvl_cdip():
    print("⏳ HuggingFace에서 RVL-CDIP 데이터셋 메타데이터 불러오는 중... (잠시만 기다려주세요)")
    
    # 1. Streaming 모드로 데이터셋 로드 (다운로드 X, 실시간 연결 O)
    dataset = load_dataset("rvl_cdip", split="train", streaming=True,trust_remote_code=True)
    print("데이터셋 연결 성공!")
    print("-" * 50)

    # 2. 클래스(라벨) 정보 확인
    # streaming 모드에서도 features 정보는 접근 가능합니다.
    features = dataset.features
    label_list = features['label'].names
    
    print(f"📌 총 클래스 개수: {len(label_list)}개")
    print(f"📌 클래스 목록:\n{label_list}")
    print("-" * 50)

    # 3. 첫 번째 샘플 딱 하나만 가져와서 분석 (next, iter 사용)
    sample = next(iter(dataset))
    
    image = sample['image'] # PIL 이미지 객체
    label_id = sample['label']
    label_name = label_list[label_id]

    # 4. 이미지 스펙 확인
    print(f"📸 [이미지 분석]")
    print(f" - 해상도(Size): {image.size} (Width x Height)")
    print(f" - 채널(Mode): {image.mode} (L=흑백, RGB=컬러)")
    print(f" - 객체 타입: {type(image)}")

    # 5. 데이터 구조 스냅샷 (JSON 형태)
    print("-" * 50)
    print(f"🏷️ [라벨 분석]")
    print(f" - 라벨 ID: {label_id}")
    print(f" - 라벨 이름: {label_name}")
    
    print("-" * 50)
    print("✅ 최종 데이터 구조 스냅샷:")
    print("{")
    print(f'  "image": "{image}",') # 실제론 이미지 객체
    print(f'  "label": {label_id}  # ({label_name})')
    print("}")

if __name__ == "__main__":
    inspect_rvl_cdip()