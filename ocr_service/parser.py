# ocr_service/parser.py
# Raw -> Schema 변환
from typing import List, Dict, Union
import numpy as np
from schemas.data_models import OCRLine

class OCRParser:
    @staticmethod
    def parse_raw_data(raw_result: Union[Dict, List]) -> List[OCRLine]:
        """
        OCR 엔진의 Raw Data(Dictionary 또는 List)를 표준화된 OCRLine 리스트로 변환합니다.
        """
        parsed_lines = []

        # 1. PaddleOCR 최신 포맷 (Parallel Lists 방식)
        # 구조: {'rec_texts': [], 'rec_scores': [], 'rec_polys': []}
        if isinstance(raw_result, dict) and 'rec_texts' in raw_result:
            texts = raw_result.get('rec_texts', [])
            scores = raw_result.get('rec_scores', [])
            boxes = raw_result.get('rec_polys', [])

            # 데이터 길이 불일치 방지 (가장 짧은 리스트 기준)
            min_len = min(len(texts), len(boxes))
            
            for i in range(min_len):
                text = texts[i]
                # 점수 리스트가 짧을 경우 기본값 0.99 처리
                score = scores[i] if i < len(scores) else 0.99
                
                # Numpy Array를 Python List로 변환
                poly = boxes[i]
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                
                # 다각형(Polygon) 좌표에서 Bounding Box(xmin, ymin, xmax, ymax) 추출
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]

                parsed_lines.append(OCRLine(
                    text=str(text),
                    bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    confidence=float(score)
                ))
            
            return parsed_lines

        # 2. 구버전 호환 (List of Objects 방식)
        # 구조: [[box, (text, score)], ...] 형태
        work_list = raw_result
        
        # 딕셔너리 내부에 리스트가 숨겨져 있는 경우 추출 ('dt_polys' 등)
        if isinstance(raw_result, dict):
            for key in ['dt_polys', 'ocr_result', 'res']:
                if key in raw_result and isinstance(raw_result[key], list):
                    work_list = raw_result[key]
                    break
        
        if not isinstance(work_list, list):
            return []

        for item in work_list:
            try:
                # [box, content] 형태인지 확인
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    box = item[0]
                    content = item[1]
                    
                    # 텍스트와 신뢰도 점수 분리
                    if isinstance(content, (list, tuple)):
                        text, score = content[0], content[1]
                    else:
                        text, score = str(content), 0.99
                    
                    # Bounding Box 계산
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    
                    parsed_lines.append(OCRLine(
                        text=str(text),
                        bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                        confidence=float(score)
                    ))
            except Exception:
                # 개별 라인 파싱 실패 시 건너뜀
                continue
                
        return parsed_lines