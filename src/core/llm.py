# src/core/llm.py
import os
from google import genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(" GOOGLE_API_KEY가 설정되지 않았습니다.")
        
        # 1. 클라이언트 설정
        self.client = genai.Client(api_key=self.api_key)
        
        # 2. 모델 설정
        self.model_name = "gemini-flash-latest" 
        print(f"✅ GeminiClient Ready (Model: {self.model_name})")

    def generate(self, prompt: str):
        try:
            # 3. 답변 생성
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return " AI 모델 연결에 실패했습니다."

if __name__ == "__main__":
    client = GeminiClient()
    print("\n🤖 질문: 안녕? 너는 누구니?")
    answer = client.generate("안녕? 너는 누구니? 짧게 대답해줘.")
    print(f"🗣️ 답변: {answer}")