import os
import json
from pathlib import Path


def get_data_pairs(raw_root, ocr_root):
    """
    이미지 폴더와 OCR 결과 폴더를 순회하며 짝을 맞춥니다.
    
    Args:
        raw_root (str): 원본 이미지 루트 (예: data/raw)
        ocr_root (str): OCR JSON 루트 (예: data/processed/ocr)
        
    Returns:
        list: [{'image_path': str, 'json_path': str, 'label': str}, ...]
    """
    data_pairs = []
    skipped_count = 0
    
    # 1. raw_root 하위의 모든 클래스 폴더 탐색
    # 예: data/raw/invoice, data/raw/resume ...
    classes = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    
    print(f"총 {len(classes)}개의 클래스 발견: {classes}")

    for cls_name in classes:
        raw_class_dir = os.path.join(raw_root, cls_name)
        ocr_class_dir = os.path.join(ocr_root, cls_name)
        
        # 해당 클래스의 OCR 폴더가 없으면 건너뜀
        if not os.path.exists(ocr_class_dir):
            print(f"[Warning] OCR 폴더 없음: {ocr_class_dir}")
            continue

        # 이미지 파일 스캔
        for img_name in os.listdir(raw_class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(raw_class_dir, img_name)
            
            # 2. 대응되는 JSON 파일 경로 추론
            # 이미지: doc_001.png -> JSON: doc_001.json
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(ocr_class_dir, json_name)
            
            # 3. JSON 존재 여부 확인 (짝이 맞는지)
            if os.path.exists(json_path):
                data_pairs.append({
                    "image_path": img_path,
                    "json_path": json_path,
                    "label": cls_name
                })
            else:
                # OCR 처리가 안 된 이미지는 스킵
                skipped_count += 1
                
    print(f"매핑 완료: 총 {len(data_pairs)}개 데이터 확보")
    if skipped_count > 0:
        print(f"건너뛴 이미지 (JSON 없음): {skipped_count}개")
        
    return data_pairs

def normalize_box(box, width, height):
    """
    PaddleOCR 좌표를 LayoutLM 포맷(0~1000)으로 변환하고,
    [x_min, y_min, x_max, y_max] 순서를 보장합니다.
    """
    # 1. 스케일링 (0~1000)
    x1 = int(box[0] / width * 1000)
    y1 = int(box[1] / height * 1000)
    x2 = int(box[2] / width * 1000)
    y2 = int(box[3] / height * 1000)

    # 2. 값 범위 고정 (Clamp 0~1000)
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))

    # 3. 좌표 정렬
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]