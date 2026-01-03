# scripts/test_text_rag.py
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.text_rag import TextRAG

def run_test():
    rag = TextRAG()

    # 상황 1: 가상의 검색 결과 (Retrieved Documents)
    mock_docs = [
        {
            "content": "2024년 1분기 마케팅 예산은 총 3억 원으로 책정되었다. 주요 지출 항목은 소셜 미디어 광고이다.",
            "metadata": {"source": "budget_2024_Q1.xlsx"}
        },
        {
            "content": "김철수 선임 연구원은 신규 AI 프로젝트의 팀장으로 임명되었다. 프로젝트 시작일은 5월 1일이다.",
            "metadata": {"source": "hr_announcement.pdf"}
        }
    ]

    print("Text RAG Test Start...\n")

    # 테스트 케이스 1: 문서에 있는 내용 질문
    q1 = "이번 분기 마케팅 예산 얼마야?"
    print(f"Q: {q1}")
    print(f"A: {rag.answer(q1, mock_docs)}")
    print("-" * 30)

    # 테스트 케이스 2: 문서에 있는 내용 질문 (다른 문서)
    q2 = "김철수 연구원은 무슨 역할을 맡았어?"
    print(f"Q: {q2}")
    print(f"A: {rag.answer(q2, mock_docs)}")
    print("-" * 30)

    # 테스트 케이스 3: 문서에 없는 내용 (Hallucination 방지 확인)
    q3 = "다음 달 회식 장소는 어디야?"
    print(f"Q: {q3}")
    print(f"A: {rag.answer(q3, mock_docs)}") # "모른다"고 해야 정답
    print("-" * 30)

    # 테스트 케이스 4: 빈 검색 결과 (Fallback 확인)
    q4 = "그냥 아무 말이나 해봐"
    print(f"Q: {q4}")
    print(f"A: {rag.answer(q4, [])}") # "검색 결과 없음" 떠야 함
    print("-" * 30)

if __name__ == "__main__":
    run_test()