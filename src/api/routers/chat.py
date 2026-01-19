# src/api/routers/chat.py
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.core.router import IntentRouter
from src.rag.multimodal_rag import MultimodalRAG
from src.rag.upload_processor import DocumentProcessor 
from src.api.schemas import ChatRequest, ChatResponse

router = APIRouter()

# 1. 엔진 로드
intent_router = IntentRouter()
rag_system = MultimodalRAG()
doc_processor = DocumentProcessor() # LayoutLM + OCR

# 2. 세션 저장소
session_store = {}
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    [업로드] LayoutLM으로 문서 종류(label)만 알아내서 세션에 저장
    """
    print(f"\n📥 [Upload] 파일 수신: {file.filename} ({session_id})")

    try:
        # 파일 저장
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # LayoutLM 분석 (오래 걸리면 비동기로 뺄 수 있음)
        processed_data = doc_processor.process_file(file_path)
        
        if not processed_data:
            return {"message": "문서 분석 실패"}

        # 세션에 정보 저장 (경로 + 라벨)
        if session_id not in session_store:
            session_store[session_id] = {"history": [], "active_file": None, "label": None}
            
        session_store[session_id]["active_file"] = file_path
        session_store[session_id]["label"] = processed_data['label'] # 예: "invoice"
        
        return {
            "message": f"분석 완료! 문서는 '{processed_data['label']}' 입니다.",
            "filename": file.filename,
            "label": processed_data['label']
        }

    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    [채팅]
    1. 파일이 있으면 -> 라우터 건너뛰고, 질문에 '[문서타입]' 정보를 붙여서 보냄.
    2. 파일이 없으면 -> 라우터 쓰고, DB 검색.
    """
    user_id = request.session_id
    
    if user_id not in session_store:
        session_store[user_id] = {"history": [], "active_file": None, "label": None}
    
    session = session_store[user_id]
    current_file = session["active_file"]
    doc_label = session["label"]
    
    print(f"\n=== Req: {request.query} [File: {os.path.basename(current_file) if current_file else 'None'}] ===")
    
    try:
        # -------------------------------------------------------
        # [로직 분기] 업로드 파일 유무에 따라 결정
        # -------------------------------------------------------
        if current_file and doc_label:
            # [Case A] 업로드 파일 있음 (Router Skip)
            print(f"🚀 [Direct] 업로드된 '{doc_label}' 문서 사용")
            
            search_category = None
            reason_msg = f"Uploaded ({doc_label})"
            
            # ★ 꿀팁: 프롬프트나 함수 수정 없이, 질문 자체에 정보를 태워서 보냄
            final_query = f"(문서 유형: {doc_label}) {request.query}"

        else:
            # [Case B] 파일 없음 -> 검색 필요 (Router Use)
            route_result = intent_router.route(request.query)
            
            search_category = route_result['filter']
            reason_msg = route_result['reason']
            final_query = request.query # 질문 그대로
            
            print(f"🤖 [Router] 검색 카테고리: {search_category}")


        # -------------------------------------------------------
        # RAG 실행
        # -------------------------------------------------------
        # multimodal.py를 수정할 필요 없이 기존 함수 그대로 호출
        answer, used_file = rag_system.answer(
            query=final_query,                # 수정된 질문 전달
            category=search_category,         
            history=session["history"][-6:], 
            target_file_path=current_file     # 파일 경로 (있으면 고정, 없으면 None)
        )
        
        # [Lock] 검색으로 파일을 찾았다면 고정 (라벨은 모름)
        if session["active_file"] is None and used_file:
            session["active_file"] = used_file
            session["label"] = "Search Result"
            print(f"📌 [Lock] 검색된 파일로 세션 고정: {os.path.basename(used_file)}")

        # 히스토리 업데이트 (저장은 원래 질문으로)
        session["history"].append(f"User: {request.query}")
        session["history"].append(f"AI: {answer}")
        
        return ChatResponse(
            response=answer,
            category=session["label"] if session["label"] else "General",
            reason=reason_msg
        )

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/chat/session/{session_id}")
async def reset_session(session_id: str):
    # 세션 초기화 (대화 기록 및 파일 고정 삭제)
    if session_id in session_store:
        del session_store[session_id]
        return {"message": "{session_id} 세션이 초기화되었습니다."}
    return {"message": "세션을 찾을 수 없습니다."}