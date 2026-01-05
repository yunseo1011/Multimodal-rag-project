# scripts/test_search.py

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.retriever import Retriever  # Gemini 기반의 새로운 검색기

def main():
    print("🚀 Gemini Embedding Search Test...")
    
    # 1. 엔진 초기화 (text-embedding-004 모델 사용)
    engine = Retriever()
    
    query = "Total amount due" # 송장에 자주 나오는 단어

    # --- TEST 1: 필터 없이 검색 ---
    # 과거 여기서 'File Folder'가 1등으로 나왔었습니다. (이미지 때문에)
    print(f"\n🔎 [TEST 1] General Search (No Filter): '{query}'")
    results = engine.retrieve(query, top_k=3)
    
    if not results:
        print("   결과 없음")
    else:
        for i, res in enumerate(results):
            label = res['metadata'].get('label', 'Unknown')
            print(f"   [{i+1}] Label: {label} | Distance: {res['distance']:.4f}")
            print(f"       Text: {res['text'][:60]}...") 

    # --- TEST 2: 필터 적용 검색 ---
    print(f"\n🔎 [TEST 2] Filtered Search (Label='invoice'): '{query}'")
    results = engine.retrieve(query, top_k=3, category="invoice")
    
    if not results:
        print("   결과 없음")
    else:
        for i, res in enumerate(results):
            label = res['metadata'].get('label', 'Unknown')
            print(f"   [{i+1}] Label: {label} | Distance: {res['distance']:.4f}")

if __name__ == "__main__":
    main()