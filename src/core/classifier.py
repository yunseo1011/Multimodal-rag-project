# src/core/classifier.py
import torch
import warnings
import os
from PIL import Image
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

from src.utils.geometry import normalize_bbox 

warnings.filterwarnings("ignore")

class DocumentClassifier:
    def __init__(self, model_path="models/layoutlmv3_finetuned.pt"):
        # 1. Device 설정
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"
            
        print(f"🔄 분류기 초기화 (Device: {self.device})")

        # 2. 클래스 정의 
        self.classes = [
            'advertisement', 'budget', 'email', 'file folder', 'form', 
            'handwritten', 'invoice', 'letter', 'memo', 'news article', 
            'presentation', 'questionnaire', 'resume', 'scientific publication', 
            'scientific report', 'specification'
        ]
        
        # 3. 모델 로드
        try:
            # Processor
            if os.path.exists("models/processor"):
                self.processor = LayoutLMv3Processor.from_pretrained("models/processor", apply_ocr=False)
            else:
                self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

            # Model
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                "microsoft/layoutlmv3-base", num_labels=len(self.classes)
            )
            
            # 가중치 로드
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"📂 Custom Model Loaded: {model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
            self.model = None

    def predict(self, image_path, ocr_result):
        """
        Args:
            image_path: 이미지 파일 경로
            ocr_result: ocr_service.aggregator가 리턴한 OCRResult 객체
        """
        if not self.model: return {"label": "error", "confidence": 0.0}
        
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # [Step 1] LayoutLM 입력 포맷으로 변환
            words = []
            boxes = []
            
            for line in ocr_result.lines:
                words.append(line.text)
                box = normalize_bbox(line.bbox, width, height)
                
                # 안전장치: 0~1000 범위를 벗어나면 모델이 에러를 뱉으므로 Clamp
                box = [max(0, min(1000, x)) for x in box]
                boxes.append(box)
            
            # 빈 문서 처리
            if not words:
                words = [" "]
                boxes = [[0, 0, 0, 0]]

            # [Step 2] 모델 추론
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            inputs = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(-1)
                idx = probs.argmax().item()
                conf = probs.max().item()

            return {"label": self.classes[idx], "confidence": round(conf, 4)}
            
        except Exception as e:
            print(f"⚠️ Prediction Error: {e}")
            return {"label": "error", "confidence": 0.0}