# schemas/data_models.py
# Pydantic 모델을 이용해서 OCR 결과의 표준 데이터 구조를 정의
from typing import List, Optional
from pydantic import BaseModel, Field

# 1. 텍스트 라인 하나에 대한 정의
class OCRLine(BaseModel):
    text: str
    bbox: List[int] = Field(..., description="[x_min, y_min, x_max, y_max]")
    confidence: float

# 2. 이미지 메타데이터
class OCRMetadata(BaseModel):
    file_name: str
    image_width: int
    image_height: int

# 3. 최종 결과 구조
class OCRResult(BaseModel):
    page_id: int = 1
    metadata: OCRMetadata
    lines: List[OCRLine]
    full_text: str = "" # 전체 텍스트를 한 문자열로 모아둔 필드 (초기값은 빈 문자열).

    def update_full_text(self):
        self.full_text = "\n".join([line.text for line in self.lines])