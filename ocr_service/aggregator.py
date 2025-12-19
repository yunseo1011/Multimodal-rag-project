import os
from PIL import Image
from schemas.data_models import OCRResult, OCRMetadata, OCRLine
from .engine import OCREngine
from .parser import OCRParser


class OCRAggregator:
    def __init__(self):
        self.engine = OCREngine()
        self.parser = OCRParser()

    def run(
        self,
        img_path: str,
        confidence_threshold: float = 0.5,
        row_tolerance: int = 15,
    ) -> OCRResult:
        # 1. 이미지 메타데이터
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # 2. OCR 실행
        raw = self.engine.extract(img_path)

        # 3. 파싱 (raw → OCRLine)
        lines = self.parser.parse_raw_data(raw)

        # 4. 검증 + 정렬 
        cleaned_lines = self._validate_and_sort_lines(
            lines=lines,
            threshold=confidence_threshold,
            img_w=img_w,
            img_h=img_h,
            row_tolerance=row_tolerance,
        )

        # 5. 결과 조립
        result = OCRResult(
            metadata=OCRMetadata(
                file_name=os.path.basename(img_path),
                image_width=img_w,
                image_height=img_h,
            ),
            lines=cleaned_lines,
        )

        # 정렬 이후에 실행
        result.update_full_text()

        return result

    def _validate_and_sort_lines(
        self,
        lines: list[OCRLine],
        threshold: float,
        img_w: int,
        img_h: int,
        row_tolerance: int,
    ) -> list[OCRLine]:
        """
        1. confidence threshold
        2. bbox sanity check + clamp
        3. 읽는 순서(좌상 → 우하) 정렬
        """
        valid_lines: list[OCRLine] = []

        for line in lines:
            # 1️.confidence 검증
            if line.confidence is None or line.confidence < threshold:
                continue

            x_min, y_min, x_max, y_max = line.bbox

            # 2️. bbox clamp (살짝 벗어난 경우 보정)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            # 완전히 잘못된 bbox 제거
            if x_min >= x_max or y_min >= y_max:
                continue

            line.bbox = [x_min, y_min, x_max, y_max]
            valid_lines.append(line)

        # 3️. 읽기 순서 정렬
        def sort_key(line: OCRLine):
            # y좌표를 tolerance 단위로 묶어서 같은 행으로 취급
            y_group = round(line.bbox[1] / row_tolerance) * row_tolerance
            return (y_group, line.bbox[0])

        valid_lines.sort(key=sort_key)

        return valid_lines

    def save_to_json(self, result: OCRResult, output_dir: str):
        """OCR 결과를 문서 단위 JSON으로 저장"""
        os.makedirs(output_dir, exist_ok=True)

        json_name = os.path.splitext(result.metadata.file_name)[0] + ".json"
        save_path = os.path.join(output_dir, json_name)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                result.model_dump_json(
                    indent=2,
                    ensure_ascii=False,  # 한글 가독성
                )
            )

        print(f" JSON Saved: {save_path}")
