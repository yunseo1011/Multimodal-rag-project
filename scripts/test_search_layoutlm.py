import sys
import os

# 프로젝트 루트 경로 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.core.retriever import SearchEngine

def main():
    engine = SearchEngine()
    
    # 1. 그냥 검색 
    print("\n 1. General Search (No Filter): 'invoice total amount'")
    results = engine.search("invoice total amount", top_k=3)
    for res in results:
        print(f"   [{res['rank']}] {res['label']} | {res['score']:.4f}")

    print("-" * 30)

    # 2. 필터 적용 검색 
    # "label이 'invoice'인 것들 중에서만 찾아라"
    print("\n 2. Filtered Search: 'invoice total amount' (Label='invoice')")
    results = engine.search("invoice total amount", top_k=3, filter_label="invoice")
    
    for res in results:
        print(f"   [{res['rank']}] {res['label']} | {res['score']:.4f}")
if __name__ == "__main__":
    main()