import requests
import json
import time

URL = "http://localhost:8000/api/v1/search"

def test_search(query_text):
    print(f"\n 질문: '{query_text}' 검색 중...")
    
    start_time = time.time()
    try:
        # API에 요청 보내기
        response = requests.post(URL, json={"query": query_text, "top_k": 3})
        
        # 응답 시간 계산
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f" 성공! (소요 시간: {elapsed:.2f}초)")
            
            # 결과 출력
            for idx, item in enumerate(results):
                print(f"  [{idx+1}] {item['label']} (유사도: {item['score']:.4f})")
                # OCR 텍스트가 너무 길면 앞부분만 출력
                print(f"      미리보기: {item['text'][:50]}...")
        else:
            print(f"실패 (Status: {response.status_code})")
            print(response.text)
            
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    # 테스트하고 싶은 질문들을 여기에 적으세요
    test_search("표 데이터 보여줘")
    test_search("invoice total amount")
    test_search("그래프 이미지")