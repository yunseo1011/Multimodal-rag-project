import torch
from src.core.model_loader import DEVICE

def extract_embedding(model, processor, ocr_result):
    """
    가공된 OCR 데이터(ocr_result)를 받아서 임베딩 벡터를 반환합니다.
    """
    # 1. 모델 입력 변환 (Tokenization)
    encoding = processor(
        ocr_result["image"],
        ocr_result["words"],
        boxes=ocr_result["boxes"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    
    # GPU 이동
    inputs = {
        "input_ids": encoding["input_ids"].to(DEVICE),
        "attention_mask": encoding["attention_mask"].to(DEVICE),
        "bbox": encoding["bbox"].to(DEVICE),
        "pixel_values": encoding["pixel_values"].to(DEVICE)
    }

    # 2. 추론 (Inference)
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 3. 텍스트 부분 슬라이싱 
        # 692(Text+Img) -> 495(Text Only)
        text_len = inputs["input_ids"].shape[1]
        text_embeddings = outputs.last_hidden_state[:, :text_len, :]
        
        # 4. Mean Pooling
        # 마스크(attention_mask)를 고려하여 평균 계산
        mask = inputs["attention_mask"].unsqueeze(-1).expand(text_embeddings.size()).float()
        sum_embeddings = torch.sum(text_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

    return embedding[0].cpu().tolist()