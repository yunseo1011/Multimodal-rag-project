# src/core/router.py
import json
import re
from src.core.llm import GeminiClient

class IntentRouter:
    def __init__(self):
        self.client = GeminiClient()
        # 헷갈릴 때는 과감하게 포기 
        self.threshold = 0.5 

    def _clean_json_text(self, text: str) -> str:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()

    def route(self, query: str):
        # RVL-CDIP 데이터셋의 16개 카테고리 정의
        prompt = f"""
        당신은 문서 검색 시스템의 '의도 분류기(Intent Classifier)'입니다.
        사용자의 질문을 분석하여 아래 16개 문서 카테고리 중 가장 적합한 하나로 분류하세요.
        
        [분류 카테고리]
        1. advertisement: 광고, 프로모션, 전단지.
        2. budget: 예산안, 예산 계획, 엑셀 시트 형태의 재무 표.
        3. email: 이메일 본문, 수신/발신 내역, 서신.
        4. file folder: 파일 폴더 표지 (내용보다는 분류용 표지).
        5. form: 신청서, 양식, 빈 칸이 많은 공문서.
        6. handwritten: 손글씨 노트, 육필 메모.
        7. invoice: 청구서, 영수증, 세금 계산서, 지불 내역.
        8. letter: 편지, 공식 서신 (우편물 형태).
        9. memo: 업무 메모, 비망록, 회의록, 사내 공지.
        10. news article: 뉴스 기사, 신문 스크랩, 컬럼.
        11. presentation: 프레젠테이션 슬라이드, PPT 자료.
        12. questionnaire: 설문지, 질문지, 퀴즈.
        13. resume: 이력서, CV, 자기소개서, 프로필.
        14. scientific publication: 학술지, 논문 출판물.
        15. scientific report: 과학/기술 보고서, 실험 리포트.
        16. specification: 사양서, 설명서, 매뉴얼, 기술 명세.
        
        17. unknown: 
           - 인사말 (안녕, 반가워) 등.
           - 위 카테고리에 명확히 속하지 않는 경우.
           - "모든 문서", "작년 자료" 처럼 범위가 너무 넓은 경우.

        [중요 규칙]
        질문자가 특정 형식을 명시하지 않았거나(예: "그냥 김철수 관련 거 다 줘"), 
        의도가 모호하면 반드시 "unknown"으로 분류하세요. 
        추측하지 마세요.

        질문: "{query}"

        [출력 형식]
        반드시 아래 JSON 형식으로만 응답하세요:
        {{
            "label": "위 카테고리 영문명 중 하나 (소문자)",
            "confidence": 0.0 ~ 1.0
        }}
        """

        try:
            raw_response = self.client.generate(prompt)
            cleaned_response = self._clean_json_text(raw_response)
            result = json.loads(cleaned_response)
            
            label = result.get("label", "unknown")
            confidence = result.get("confidence", 0.0)

            # unknown이거나 확신이 없으면 필터 해제
            if label == "unknown" or confidence < self.threshold:
                final_filter = None
                reason = "포괄적 질문 또는 의도 불명 -> 전체 검색"
            else:
                final_filter = label
                reason = f"카테고리 감지됨 ({label}, {confidence})"

            return {
                "query": query,
                "label": label,
                "confidence": confidence,
                "filter": final_filter,
                "reason": reason
            }

        except Exception as e:
            print(f"⚠️ Router Error: {e}")
            return {"query": query, "label": "error", "confidence": 0.0, "filter": None, "reason": "System Error"}