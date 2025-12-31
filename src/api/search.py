from fastapi import APIRouter, HTTPException
from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from src.core.retriever import SearchEngine

router = APIRouter()

print("🚀 Loading Search Engine for API...")
search_engine = SearchEngine()

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Semantic Search Endpoint
    - query: 질문 텍스트
    - top_k: 반환할 문서 수
    - filter_label: (선택) 특정 카테고리 필터링
    """
    try:
        # 1. 검색 엔진 호출 (필터가 있으면 전달)
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            filter_label=request.filter_label
        )
        
        # 2. 결과 변환 (Dict -> Pydantic Schema)
        response_items = []
        for res in results:
            response_items.append(SearchResultItem(
                rank=res['rank'],
                doc_id=res['id'],
                score=res['score'],
                label=res['label'],
                file_path=res['file_path'],
                text=res['preview']
            ))
            
        return SearchResponse(results=response_items)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))