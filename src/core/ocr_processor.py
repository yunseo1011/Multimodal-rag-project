import json
from PIL import Image
from src.utils.geometry import normalize_bbox

def process_ocr_data(image_path, json_path):
    """
    이미지와 JSON 경로를 받아서, 모델에 들어갈 형태(words, boxes)로 가공합니다.
    """
    # 1. 이미지 로드
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 2. JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    # 3. 데이터 가공 
    items = []
    
    if isinstance(ocr_data, list):
        # 전체가 리스트면 그대로 사용
        items = ocr_data
    elif isinstance(ocr_data, dict):
        # 딕셔너리라면, 값들 중에서 "리스트인 것만" 골라서 합치기
        # 예: "form": [...], "lines": [...] 같은 것만 가져오고, "version": 1 은 무시
        for value in ocr_data.values():
            if isinstance(value, list):
                items.extend(value)

    words = []
    boxes = []

    for item in items:
        # 딕셔너리인지 확인
        if not isinstance(item, dict):
            continue
            
        # 키 이름이 제각각일 수 있어서 여러 개 체크
        text = item.get("text", "") or item.get("words", "")
        box = item.get("box", []) or item.get("bbox", [])
        
        # 텍스트가 비었거나, 박스 좌표가 4개가 아니면 건너뜀
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
            
        words.append(text)
        boxes.append(normalize_bbox(box, width, height))

    return {
        "image": image,
        "words": words,
        "boxes": boxes
    }