import sys
import os
from dotenv import load_dotenv

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.router import IntentRouter  
from src.rag.multimodal_rag import MultimodalRAG

load_dotenv()

def main():

    # 1. 시스템 초기화
    router = IntentRouter()
    rag = MultimodalRAG()

    # 2. 테스트할 질문 리스트
    test_questions = [
        "이 영수증 합계 금액이 얼마야?",          # 예상: invoice
        "마케팅 예산안 좀 찾아줘",               # 예상: budget
        "안녕 반가워, 넌 누구니?",               # 예상: unknown (전체검색 or 대화)
        "이 문서에 서명이나 도장이 찍혀있어?",     # 예상: form 또는 unknown
        "손글씨로 적힌 메모 내용 읽어줘"           # 예상: handwritten
    ]

    # 3. 루프 돌면서 테스트
    for i, query in enumerate(test_questions):
        print(f"\n [질문 {i+1}] {query}")
        
        # Router 단계 
        route_result = router.route(query)
        category = route_result['filter']
        reason = route_result['reason']
        
        print(f"    [Router 판단] 카테고리: {category if category else '전체(None)'}")
        print(f"      └ 이유: {reason}")

        # RAG 단계 
        answer = rag.answer(query, category=category)
        
        print(f"   🤖 [Gemini 답변]\n   {answer}")
        print("-" * 70)

if __name__ == "__main__":
    main()