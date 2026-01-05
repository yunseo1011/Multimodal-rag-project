# src/core/embedding.py
'''
기존: LayoutLM 모델을 로드해서 이미지 넣고 추출 
변경: 구글 Gemini에게 텍스트만 넣고 벡터 받아오는 코드로 변경.
'''
import os
import time
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 로드 (GOOGLE_API_KEY)
load_dotenv()

# Gemini 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding(text: str, retries: int = 3) -> List[float]:
    """
    텍스트를 입력받아 Gemini의 의미 기반 벡터(768차원)를 반환
    (검색/RAG용 모델: text-embedding-004)
    """
    # 텍스트가 너무 짧거나 없으면 빈 벡터 방지
    if not text or len(text.strip()) < 2:
        return [0.0] * 768

    for attempt in range(retries):
        try:
            # Gemini 임베딩 요청
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document", # 문서를 DB에 저장할 때 쓰는 모드
                title=None
            )
            return result['embedding']
        except Exception as e:
            print(f" 임베딩 요청 실패 ({attempt+1}/{retries}): {e}")
            time.sleep(1) # 1초 대기 후 재시도
            
    print(" 최종 실패: 임베딩 생성 불가")
    return [0.0] * 768

if __name__ == "__main__":
    vec = get_embedding("삼성전자 이번 달 청구서입니다.")
    print(f"✅ 벡터 생성 완료! 차원 수: {len(vec)}") # 768 나와야 함
    print(f"   데이터 예시: {vec[:5]}...")