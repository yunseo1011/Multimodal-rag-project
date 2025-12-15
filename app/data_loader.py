# app/data_loader.py
from datasets import load_dataset
from PIL import Image

# 1. 데이터셋 연결 
print("⏳ 데이터셋 스트리밍 연결 중... (최초 1회만 느림)")
dataset = load_dataset("rvl_cdip", split="train", streaming=True, trust_remote_code=True)

# 2. 데이터 파이프(Iterator)를 미리 만들어둡니다.

data_iterator = iter(dataset)

def get_random_document():
    global data_iterator
    
    print("⚡️ 이미지 가져오는 중...")
    
    try:
        # 3. 섞지 않고 '다음 순서'의 이미지를 즉시 가져옵니다.
        # 이미 파이프가 연결돼 있어서 훨씬 빠릅니다.
        sample = next(data_iterator)
    except StopIteration:
        # 혹시 데이터가 바닥나면(그럴 일 없지만) 다시 연결
        data_iterator = iter(dataset)
        sample = next(data_iterator)
    
    image = sample['image']
    label_id = sample['label']
    
    # 라벨 이름 찾기
    label_names = dataset.features['label'].names
    label_name = label_names[label_id]
    
    # 흑백 -> RGB 변환
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    return image, label_name