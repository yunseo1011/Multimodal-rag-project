import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.core.retriever import SearchEngine

def print_results(results):
    """결과를 보기 좋게 출력하는 헬퍼 함수"""
    if not results:
        print("   ❌ 검색 결과가 없습니다.")
        return
        
    for res in results:
        # 파일 경로에서 파일명만 깔끔하게 추출
        filename = os.path.basename(res.get('file_path', 'Unknown'))
        
        print(f"   [{res['rank']}] Label: {res['label']:<10} | "
              f"Score: {res['score']:.4f} | "
              f"File: {filename}")

def main():
    engine = SearchEngine()
    
    print("=" * 60)
    
    # ---------------------------------------------------------
    # TEST 1: 일반 검색 (우리가 아는 그 문제 상황)
    # ---------------------------------------------------------
    # 설명: 이미지(검은색) 페널티 때문에 'File Folder'가 나올 가능성 높음
    print("\n🔍 1. General Search (No Filter): 'invoice total amount'")
    results = engine.search("invoice total amount", top_k=10)
    print_results(results)

    # ---------------------------------------------------------
    # TEST 2: 필터 적용 검색 (우리의 해결책)
    # ---------------------------------------------------------
    # 설명: 'invoice' 라벨 안에서 찾으므로 정확한 송장이 나와야 함
    print("\n🔍 2. Filtered Search (Label='invoice'): 'invoice total amount'")
    results = engine.search("invoice total amount", top_k=3, filter_label="invoice")
    print_results(results)

    print("-" * 30)

    # ---------------------------------------------------------
    # TEST 3: 지능 검증 - 이력서 찾기 (필터 없음)
    # ---------------------------------------------------------
    # 설명: 필터가 없어도 'Education', 'Python'을 보고 'Resume'를 찾아야 함
    # -> 이게 성공하면 "랜덤이 아니다"라는 확실한 증거!
    print("\n🧠 3. Intelligence Check (No Filter): 'Education Experience Python'")
    results = engine.search("Education Experience Python", top_k=10)
    print_results(results)

    # ---------------------------------------------------------
    # TEST 4: 지능 검증 - 강력한 송장 키워드 (필터 없음)
    # ---------------------------------------------------------
    # 설명: 'Bill to Ship to'는 송장에만 있는 아주 강력한 단어임.
    # -> 텍스트 힘이 강하면 이미지 페널티를 이기고 Invoice가 나올 수도 있음!
    print("\n🧪 4. Strong Pattern Check (No Filter): 'Bill to Ship to Payment terms'")
    results = engine.search("Bill to Ship to Payment terms", top_k=10)
    print_results(results)

    print("=" * 60)

if __name__ == "__main__":
    main()