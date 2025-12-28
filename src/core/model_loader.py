import os
import torch
from transformers import LayoutLMv3Model, LayoutLMv3Processor

# 싱글톤 인스턴스
_MODEL = None
_PROCESSOR = None
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. 현재 파일 위치: src/core/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 2. 프로젝트 루트: src/core/ -> src/ -> Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 

# 모델이 'models' 폴더 안에 있다고 알려줌
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "layoutlmv3_finetuned.pt")

def get_model(model_path=None):
    global _MODEL, _PROCESSOR
    
    # 경로가 안 들어오면 위에서 설정한 기본 경로 사용
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # 이미 로딩했으면 리턴 (Singleton)
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        fallback_path = os.path.join(PROJECT_ROOT, "layoutlmv3_finetuned.pt")
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            raise FileNotFoundError(
                f"\n❌ [ERROR] 모델 파일을 찾을 수 없습니다!\n"
                f"1순위 검색: {model_path}\n"
                f"2순위 검색: {fallback_path}\n"
                f"확인: 'models' 폴더 안에 'layoutlmv3_finetuned.pt' 파일이 있는지 봐주세요."
            )

    print(f"🔄 Loading Model from: {model_path}")
    print(f"   Device: {DEVICE}")
    
    # 1. 프로세서 로딩
    _PROCESSOR = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    # 2. 모델 로딩
    model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
    
    # 3. 가중치 로드
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        if "classifier" in key: continue
        if key.startswith("layoutlmv3."):
            new_state_dict[key.replace("layoutlmv3.", "")] = value
        else:
            new_state_dict[key] = value
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    _MODEL = model
    print("Model Loaded Successfully!")
    
    return _MODEL, _PROCESSOR