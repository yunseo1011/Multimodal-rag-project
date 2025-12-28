from fastapi import APIRouter
from src.core.model_loader import get_model

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    # 모델이 메모리에 있는지 확인
    try:
        model, _ = get_model()
        is_loaded = model is not None
    except:
        is_loaded = False
        
    return {
        "status": "ok", 
        "model_loaded": is_loaded
    }