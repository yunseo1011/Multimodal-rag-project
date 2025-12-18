# ocr_service/engine.py
# 클래스로 설계해 인스턴스를 메모리에 한번만 올리도록. 
# API 호출할때마다 모델 로딩하면 느림

import os
from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self, lang: str = 'en'):
        print("Loading PaddleOCR model... ")
        # 인스턴스 생성 시 한 번만 로딩
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang=lang
        )
        print(" Model loaded.")

    def extract(self, img_path: str):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        result = self.ocr.ocr(img_path)
        if not result or result[0] is None:
            return []
        return result[0]