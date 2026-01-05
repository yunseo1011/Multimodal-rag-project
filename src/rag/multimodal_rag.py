import os
from PIL import Image
from src.core.llm import GeminiClient
from src.rag.retriever import Retriever
from src.rag.text_rag import TextRAG
from src.rag.prompts import VISION_RAG_PROMPT 

class MultimodalRAG:
    def __init__(self):
        self.llm = GeminiClient()
        self.retriever = Retriever()
        self.text_rag = TextRAG()

    def answer(self, query: str, category: str = None):
        retrieved_docs = self.retriever.retrieve(query, top_k=3, category=category)
        if not retrieved_docs:
            return "검색 결과가 없어 답변할 수 없습니다."

        top_doc = retrieved_docs[0]
        file_path = top_doc['metadata']['file_path']
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.png', '.jpg', '.jpeg']: # 데이터셋은 다 이미지에 해당
            print(f"🖼️ [Vision Mode] 이미지 발견! ({os.path.basename(file_path)})")
            return self._handle_image_query(query, file_path)
        else:
            print(f"📝 [Text Mode] 텍스트 문서 발견!") # 사용 x. 확장 시 사용 가능
            return self.text_rag.answer(query, retrieved_docs)

    def _handle_image_query(self, query: str, image_path: str):
        try:
            if not os.path.exists(image_path):
                return f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}"

            img = Image.open(image_path)
            file_name = os.path.basename(image_path)

            # 2. 프롬프트에 질문과 파일명을 끼워 넣습니다.
            final_prompt = VISION_RAG_PROMPT.format(
                query=query,
                file_name=file_name
            )
            
            # 3. [프롬프트(텍스트), 이미지] 리스트로 전달
            response = self.llm.generate([final_prompt, img])
            
            return response
            
        except Exception as e:
            print(f"⚠️ 이미지 처리 에러: {e}")
            return "이미지 분석 중 오류가 발생했습니다."