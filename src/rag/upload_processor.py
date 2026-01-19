# src/rag/upload_processor.py
import os
from dotenv import load_dotenv

from src.core.classifier import DocumentClassifier
from ocr_service.aggregator import OCRAggregator

load_dotenv()

class DocumentProcessor:  
    def __init__(self):
        # DB 관련 코드 싹 제거!
        
        # AI 엔진 로드
        print("🔧 [Processor] AI 엔진 로드 중...")
        self.ocr_aggregator = OCRAggregator() 
        self.classifier = DocumentClassifier() 
        print("✅ [Processor] 준비 완료.")

    def process_file(self, file_path: str):
        """
        파일을 읽어서 텍스트와 라벨을 반환합니다. (DB 저장 X)
        """
        print(f"\n📥 [Processing] 파일 분석 중: {os.path.basename(file_path)}")
        
        try:
            # 1. OCR 실행
            ocr_result = self.ocr_aggregator.run(file_path)
            full_text = ocr_result.full_text
            
            if not full_text.strip():
                return None

            # 2. 문서 분류 (LayoutLM)
            cls_res = self.classifier.predict(file_path, ocr_result)
            label = cls_res['label']
            confidence = cls_res['confidence']
            
            print(f"🏷️ 분류 결과: {label} ({confidence})")

            # 3. 결과 리턴 (저장하지 않고 그냥 돌려줌)
            return {
                "text": full_text,
                "label": label,
                "file_path": file_path
            }
            
        except Exception as e:
            print(f"❌ 처리 중 오류: {e}")
            return None