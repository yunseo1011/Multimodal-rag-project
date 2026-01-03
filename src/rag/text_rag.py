# src/rag/text_rag.py
from src.core.llm import GeminiClient
from src.rag.prompts import TEXT_RAG_PROMPT

class TextRAG:
    def __init__(self):
        self.llm = GeminiClient()

    def answer(self, query: str, retrieved_docs: list) -> str:
        """
        retrieved_docs: [{'content': '...', 'metadata': {'source': '...'}}, ...]
        """
        
        # 1. 검색 결과가 아예 없는 경우 (Graceful Fallback)
        if not retrieved_docs:
            return "검색 결과가 없어 답변할 수 없습니다."

        # 2. Context 구성 (문서 내용 + 출처 정보를 하나의 문자열로 합침)
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"문서 {i+1} (파일명: {source}):\n{content}")
        
        context_str = "\n\n".join(context_parts)

        # 3. 프롬프트 완성
        final_prompt = TEXT_RAG_PROMPT.format(
            context_str=context_str,
            query=query
        )

        # 4. LLM 답변 생성
        response = self.llm.generate(final_prompt)
        return response