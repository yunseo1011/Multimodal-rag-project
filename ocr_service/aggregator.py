# ocr_service/aggregator.py
# 문서 단위 JSON 생성
from PIL import Image
from schemas.data_models import OCRResult, OCRMetadata
from .engine import OCREngine
from .parser import OCRParser

class OCRAggregator:
    def __init__(self):
        self.engine = OCREngine()
        self.parser = OCRParser()

    def run(self, img_path: str) -> OCRResult:
        # 1. 이미지 메타데이터 추출
        with Image.open(img_path) as img:
            w, h = img.size
            
        # 2. OCR 실행
        raw = self.engine.extract(img_path)
        
        # 3. 파싱 (raw 데이터 -> pydantic 모델 변환)
        lines = self.parser.parse_raw_data(raw)
        
        # 4. 결과 조립
        result = OCRResult(
            metadata=OCRMetadata(
                file_name=img_path.split("/")[-1],
                image_width=w,
                image_height=h
            ),
            lines=lines
        )
        
        result.update_full_text()
        return result

# 테스트용 코드
if __name__ == "__main__":
    test_img = "data/raw/invoice/doc_0000.png"  # 예시 경로
    
    import os
    if os.path.exists(test_img):
        agg = OCRAggregator()
        res = agg.run(test_img)
        print(res.model_dump_json(indent=2))
    else:
        print(f"❌ 이미지를 찾을 수 없습니다: {test_img}")
