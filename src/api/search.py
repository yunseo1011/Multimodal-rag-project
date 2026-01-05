# src/api/search.py

from fastapi import APIRouter, HTTPException
from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from src.rag.retriever import Retriever  

router = APIRouter()

print(" Loading Gemini Retriever for API...")
retriever = Retriever() 

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Semantic Search Endpoint (Gemini Embedding)
    """
    try:
        # 1. 검색 엔진 호출
        # category 매개변수로 전달 
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            category=request.filter_label 
        )
        
        # 2. 결과 변환 (Dict -> Pydantic Schema)
        response_items = []
        
        if not results:
            return SearchResponse(results=[])

        for i, res in enumerate(results):
            # ChromaDB 구조에 맞춰 데이터 추출
            # metadata 안에 label, file_path가 들어있음
            metadata = res.get('metadata', {})
            
            response_items.append(SearchResultItem(
                rank=i + 1,
                doc_id=res.get('id', 'unknown'), # doc_id가 없다면 unknown
                score=res.get('distance', 0.0), # distance 값 사용
                label=metadata.get('label', 'N/A'),
                file_path=metadata.get('file_path', 'N/A'),
                text=res.get('text', '')[:200] # 텍스트 미리보기 (너무 길면 자름)
            ))
            
        return SearchResponse(results=response_items)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))