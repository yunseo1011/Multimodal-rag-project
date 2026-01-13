# src/api/routers/chat.py
from fastapi import APIRouter, HTTPException
from src.core.router import IntentRouter
from src.rag.multimodal_rag import MultimodalRAG
from src.api.schemas import ChatRequest, ChatResponse

router = APIRouter()
intent_router = IntentRouter()
rag_system = MultimodalRAG()

# 세션 저장소 { user_id: { history: [], locked_file: path } }
session_store = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.session_id
    print(f"\n=== Req: {request.query} ({user_id}) ===")
    
    # 1. 세션 없으면 생성
    if user_id not in session_store:
        session_store[user_id] = {"history": [], "locked_file": None}
    
    session = session_store[user_id]
    
    try:
        # 2. 의도 분류
        route_result = intent_router.route(request.query)
        
        # 3. RAG 실행 (고정된 파일이 있다면 전달)
        answer, used_file = rag_system.answer(
            query=request.query, 
            category=route_result['filter'], 
            history=session["history"][-6:], # 최근 대화 내용
            fixed_file_path=session["locked_file"]
        )
        
        # 4. 파일 고정 (처음 파일을 찾았다면 세션에 저장)
        if session["locked_file"] is None and used_file:
            session["locked_file"] = used_file

        # 5. 대화 기록 업데이트
        session["history"].append(f"User: {request.query}")
        session["history"].append(f"AI: {answer}")
        
        return ChatResponse(
            response=answer,
            category="Locked" if session["locked_file"] else "Search",
            reason=route_result['reason']
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/session/{session_id}")
async def reset_session(session_id: str):
    # 세션 초기화 (대화 기록 및 파일 고정 삭제)
    if session_id in session_store:
        del session_store[session_id]
        return {"message": "{session_id} 세션이 초기화되었습니다."}
    return {"message": "세션을 찾을 수 없습니다."}