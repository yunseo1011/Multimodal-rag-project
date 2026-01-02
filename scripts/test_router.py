# scripts/test_router.py
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.router import IntentRouter

def run_test():
    router = IntentRouter()
    
    test_queries = [
        "삼성전자 관련 뉴스 기사 좀 찾아줘",      # -> news article
        "이번 프로젝트 예산안 엑셀 파일 있어?",    # -> budget
        "김철수 연구원의 실험 보고서",            # -> scientific report
        "제품 사양서랑 매뉴얼 보여줘",            # -> specification
        "채용 지원자들 이력서 모음",              # -> resume
        "안녕? 심심하다",                        # -> unknown
        "2024년도 전체 자료 검색해줘",             # -> unknown (포괄적)
        "invoice total amount"                #   invoice
    ]

    print(" Router Full-Scale Test Start...\n")

    for i, q in enumerate(test_queries):
        # 429 에러 방지를 위해 살짝 대기 (필요 시)
        if i > 0: time.sleep(1) 
        
        result = router.route(q)
        print(f"Q: {result['query']}")
        print(f"🎯 Filter: {result['filter']}") 
        print(f"📝 Reason: {result['reason']}")
        print("-" * 30)

if __name__ == "__main__":
    run_test()