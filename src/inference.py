import os
import json
import torch
import warnings
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch.nn.functional as F

# 경고 메시지 숨기기 (깔끔한 출력을 위해)
warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 프로젝트 루트 경로 (현재 파일이 src/ 안에 있다고 가정하고 상위 폴더 지정)
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

# 레이블 맵 (학습 시 사용한 것과 순서가 동일해야 함)
LABELS = [
    'advertisement', 'budget', 'email', 'file folder', 'form', 'handwritten', 
    'invoice', 'letter', 'memo', 'news article', 'presentation', 'questionnaire', 
    'resume', 'scientific publication', 'scientific report', 'specification'
]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}

# 전역 변수로 모델/프로세서 선언 (최초 실행 시 로드)
_model = None
_processor = None

# ==========================================
# 2. 모델 로딩 (Singleton 패턴)
# ==========================================
def get_model_and_processor():
    """모델과 프로세서가 로드되어 있지 않으면 로드하고, 있으면 반환합니다."""
    global _model, _processor
    
    if _model is None:
        print(f"🔄 Loading model from {MODEL_PATH} on {DEVICE}...")
        
        # 프로세서 로드
        _processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_NAME, apply_ocr=False)
        
        # 모델 로드
        _model = LayoutLMv3ForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=len(LABELS), label2id=label2id, id2label=id2label
        )
        
        # 가중치 덮어쓰기
        if DEVICE == "cpu":
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
        else:
            state_dict = torch.load(MODEL_PATH)
            
        _model.load_state_dict(state_dict)
        _model.to(DEVICE)
        _model.eval()
        print("✅ Model loaded successfully.")
        
    return _model, _processor

# ==========================================
# 3. OCR 및 전처리 (Helper Functions)
# ==========================================
def normalize_box(box, width, height):
    """좌표를 0~1000 스케일로 정규화"""
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def run_ocr(image, json_path=None):
    """
    이미지에 대해 OCR을 수행합니다.
    1. json_path가 있고 파일이 존재하면 -> JSON 로드 (속도 빠름, 기존 데이터용)
    2. 없으면 -> Tesseract OCR 수행 (새로운 파일용) -> *4주차에 PaddleOCR로 교체 예정*
    """
    # 1. JSON 파일이 있으면 우선 로드 (재현성 확보)
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 데이터 포맷 처리 ('words' 또는 'full_text')
            words = []
            if 'words' in data:
                words = data['words']
            elif 'full_text' in data:
                words = data['full_text'].split()
            
            # BBox 정보가 JSON에 없다면 더미 박스 생성 (LayoutLMv3는 BBox 필수)
            # *실제로는 JSON에 bbox도 저장해두는 것이 좋습니다.
            # 여기서는 JSON에 텍스트만 있다고 가정할 때의 Fallback입니다.
            boxes = [[0, 0, 0, 0]] * len(words) 
            if 'bboxes' in data:
                 boxes = data['bboxes']

            return words, boxes
        except Exception as e:
            print(f"⚠️ JSON load failed, falling back to OCR engine: {e}")

    # 2. JSON이 없으면 실시간 OCR 수행 (pytesseract)
    # 4주차 Refactoring 목표: 여기서 PaddleOCR 호출로 변경
    import pytesseract
    
    ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    width, height = image.size
    
    for i, text in enumerate(ocr_df['text']):
        if text.strip() != "":
            words.append(text)
            # 원본 좌표 (left, top, width, height) -> (x1, y1, x2, y2)
            x1 = ocr_df['left'][i]
            y1 = ocr_df['top'][i]
            x2 = x1 + ocr_df['width'][i]
            y2 = y1 + ocr_df['height'][i]
            
            # 정규화 (0~1000)
            boxes.append(normalize_box([x1, y1, x2, y2], width, height))
            
    return words, boxes

# ==========================================
# 4. 메인 추론 함수 (Predict)
# ==========================================
def predict(image_path, json_path=None):
    """
    이미지 경로를 받아 분류 결과를 반환합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        json_path (str, optional): 미리 계산된 OCR JSON 파일 경로
        
    Returns:
        dict: {label, confidence, probabilities}
    """
    model, processor = get_model_and_processor()
    
    # 이미지 로드
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Image load failed: {str(e)}"}

    # OCR 수행
    words, boxes = run_ocr(image, json_path)
    
    if len(words) == 0:
        return {"error": "No text detected in image."}

    # 모델 입력 변환
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
    # 결과 해석
    top_prob, top_idx = torch.max(probs, dim=-1)
    predicted_label = id2label[top_idx.item()]
    confidence = top_prob.item()

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence * 100, 2),
        "input_words_count": len(words)
    }

# ==========================================
# 5. 실행 테스트 (CLI)
# ==========================================
if __name__ == "__main__":
    # 테스트할 이미지 경로 (아무거나 하나 지정해보세요)
    # 예: data/raw/budget/doc_0001.png
    TEST_IMAGE = "data/raw/budget/doc_0572.png"  
    
    # JSON 파일 경로 추론 (선택사항)
    # data/raw/budget/doc_0572.png -> data/processed/ocr/budget/doc_0572.json
    # TEST_JSON = TEST_IMAGE.replace("raw", "processed/ocr").replace(".png", ".json")
    
    if os.path.exists(TEST_IMAGE):
        print(f"🚀 Predicting for: {TEST_IMAGE}")
        result = predict(TEST_IMAGE)
        print("\n📊 Result:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print("Please change 'TEST_IMAGE' variable in __main__ block.")