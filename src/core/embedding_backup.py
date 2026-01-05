# src/core/embedding_backup.py
# 수정 전 코드 (layoutlm)으로 임베딩 추출
import os
import json
import torch
import warnings
from PIL import Image
from transformers import LayoutLMv3Model, LayoutLMv3Processor

# 경고 메시지 끄기
warnings.filterwarnings("ignore")

# 1. 설정 (Configuration)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/layoutlmv3_finetuned.pt")
BASE_MODEL_NAME = "microsoft/layoutlmv3-base"

# 디바이스 설정
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 2. 풀링 전략 구현
def mean_pooling(last_hidden_state, attention_mask):
    # Mask 확장 [Batch, Seq] -> [Batch, Seq, 1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    
    # Count (0으로 나누기 방지)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sum_embeddings / sum_mask

def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0, :]

# 3. 모델 로딩
def get_embedding_model():
    print(f"🔄 Loading Base Model ({BASE_MODEL_NAME}) on {DEVICE}...")
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_NAME, apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained(BASE_MODEL_NAME)
    
    if os.path.exists(MODEL_PATH):
        if DEVICE == "cpu":
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
        else:
            state_dict = torch.load(MODEL_PATH)
            
        new_state_dict = {}
        for key, value in state_dict.items():
            if "classifier" in key: continue
            if key.startswith("layoutlmv3."):
                new_key = key.replace("layoutlmv3.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Custom weights loaded.")
    else:
        print(f"⚠️ Fine-tuned model not found. Using base model.")

    model.to(DEVICE)
    model.eval()
    return model, processor

# 4. 유틸리티
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

# 5. 임베딩 추출 (강력한 디버깅 및 안전장치 추가)
def extract_embedding(model, processor, image_path, json_path, strategy="mean"):
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        words = []
        boxes = []
        
        if "lines" in data:
            for line in data["lines"]:
                text = line.get("text", "").strip()
                bbox = line.get("bbox", [])
                if text and len(bbox) == 4:
                    words.append(text)
                    boxes.append(normalize_box(bbox, width, height))
        
        if not words:
            words = [" "]
            boxes = [[0, 0, 0, 0]]

        # 1. 프로세서 호출
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 2. [강제 동기화] 가장 짧은 길이(min_len) 찾기
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        bbox = encoding["bbox"]
        
        # 현재 길이 확인
        len_ids = input_ids.shape[1]
        len_mask = attention_mask.shape[1]
        len_bbox = bbox.shape[1]
        
        # 디버깅: 길이가 다르면 출력
        if len_ids != len_mask or len_ids != len_bbox:
            print(f"⚠️ Shape Mismatch Detected! IDs:{len_ids}, Mask:{len_mask}, BBox:{len_bbox}")

        # 가장 짧은 길이로 통일 (최대 512)
        min_len = min(len_ids, len_mask, len_bbox, 512)
        
        # 3. 새로운 입력 딕셔너리 생성 (Clean Dictionary)
        # 기존 딕셔너리를 수정하지 않고, 확실하게 잘린 놈들만 담습니다.
        clean_inputs = {
            "input_ids": input_ids[:, :min_len].to(DEVICE),
            "attention_mask": attention_mask[:, :min_len].to(DEVICE),
            "bbox": bbox[:, :min_len, :].to(DEVICE),
            "pixel_values": encoding["pixel_values"].to(DEVICE)
        }

        # 4. 추론
        with torch.no_grad():
            outputs = model(**clean_inputs)
            
            # 🔍 [핵심 수정] 출력에서 텍스트 부분만 발라내기
            # 모델 출력(692) = 텍스트(495) + 이미지(197)
            # 우리는 텍스트 마스크(495)를 쓸 거니까, 출력도 앞부분 495개만 가져와야 함.
            
            text_len = clean_inputs["input_ids"].shape[1] # 예: 495
            
            # 전체 출력 중 앞부분(텍스트)만 슬라이싱
            text_embeddings = outputs.last_hidden_state[:, :text_len, :] 
            
            if strategy == "mean":
                # 이제 text_embeddings(495)와 attention_mask(495) 길이가 딱 맞음!
                embedding = mean_pooling(text_embeddings, clean_inputs["attention_mask"])
            else: 
                # CLS 토큰은 어차피 0번째라 상관없었음
                embedding = cls_pooling(text_embeddings)
                
        return embedding[0].cpu().tolist()

    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return None

# 6. 실행
if __name__ == "__main__":
    TEST_IMAGE = "data/raw/memo/doc_0047.png" 
    TEST_JSON = "data/processed/ocr/memo/doc_0047.json"
    
    if os.path.exists(TEST_IMAGE) and os.path.exists(TEST_JSON):
        model, processor = get_embedding_model()
        
        print(f"\n🧪 Extracting embedding for: {os.path.basename(TEST_IMAGE)}")
        
        # 1) CLS Pooling
        vec_cls = extract_embedding(model, processor, TEST_IMAGE, TEST_JSON, strategy="cls")
        if vec_cls:
            print(f"🔹 [CLS] Vector Dim: {len(vec_cls)}")
            print(f"   Values (First 5): {vec_cls[:5]}")
        
        # 2) Mean Pooling
        vec_mean = extract_embedding(model, processor, TEST_IMAGE, TEST_JSON, strategy="mean")
        if vec_mean:
            print(f"🔹 [Mean] Vector Dim: {len(vec_mean)}")
            print(f"   Values (First 5): {vec_mean[:5]}")
            
        print("\n✅ Verification:")
        if vec_cls and vec_mean and vec_cls != vec_mean:
            print("👉 Success! Strategies produced different vectors.")
    else:
        print("❌ Test files not found.")