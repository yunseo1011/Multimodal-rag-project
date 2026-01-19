#src/rag/multimodal_rag.py
import os
import re
from PIL import Image

from src.core.llm import GeminiClient
from src.rag.retriever import Retriever
from src.rag.text_rag import TextRAG
from src.rag.prompts import VISION_RAG_PROMPT

# Docker / Local 데이터 경로
DOCKER_DATA_DIR = "/app/data"
LOCAL_DATA_DIR = "./data"

class MultimodalRAG:
    def __init__(self):
        self.llm = GeminiClient()
        self.retriever = Retriever()

    def answer(self, query: str, category: str = None, 
               history: list = None, target_file_path: str = None):

        history_text = ""
        if history:
            history_text = "이전 대화 내역:\n" + "\n".join(history) + "\n\n"

        # 이미 고정된 파일이 들어온 경우 (업로드 or 이전 대화 고정)
        if target_file_path and os.path.exists(target_file_path):
            print(f"🔒 [Locked] 고정된 문서 분석: {os.path.basename(target_file_path)}")
            
        # 파일이 없으면 -> DB 검색 수행
        else:
            print(f"🔍 [Search] 파일 없음 -> DB 검색 수행: {query}")
            retrieved_docs = self.retriever.retrieve(query, top_k=5, category=category)
            
            if not retrieved_docs:
                return "검색 결과가 없어 답변할 수 없습니다.", None

            # Rerank로 가장 좋은 문서 하나 선정
            target_doc = self._select_best_doc(query, retrieved_docs)
            original_path = target_doc["metadata"].get("file_path", "")
            filename = target_doc["metadata"].get("filename", os.path.basename(original_path))

            # 경로 보정
            target_file_path = self._resolve_file_path(original_path, filename)
            
            if not target_file_path:
                return "파일을 찾을 수 없습니다.", None
            
            print(f"🎯 [Found] 검색된 파일: {os.path.basename(target_file_path)}")

        print(f"🖼️ [Vision] 이미지 분석 시작: {os.path.basename(target_file_path)}")
        response = self._handle_image_query(query, target_file_path, history_text)
        
        return response, target_file_path
    
    # Top-K 문서 Reranker
    def _select_best_doc(self, query, candidates):
        """
        Top-K 문서 중 '이 문서로 질문에 답할 수 있는가?' 기준으로 LLM이 최적의 문서를 선택
        """
        try:
            candidates_info = ""
            for i, doc in enumerate(candidates):
                fname = doc["metadata"].get(
                    "filename",
                    os.path.basename(doc["metadata"].get("file_path", "Unknown"))
                )

                text = doc.get("text", "")
                preview = text[:800].replace("\n", " ")

                candidates_info += f"""
                [{i+1}]
                파일명: {fname}
                내용:
                {preview}
                """

            prompt = f"""
            당신은 문서 검색 시스템의 재선별기(Reranker)입니다.

            사용자 질문:
            "{query}"

            아래 문서들 중에서
            "이 문서를 읽으면 위 질문에 답할 수 있는가"를 기준으로
            가장 적합한 하나를 고르세요.

            문서 후보:
            {candidates_info}

            규칙:
            - 질문에 답할 수 없는 문서는 고르지 마세요.
            - 가장 적합한 문서 번호 하나만 숫자로 출력하세요.
            """

            response = self.llm.generate(prompt).strip()
            match = re.search(r"\d+", response)

            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]

            # fallback
            return candidates[0]

        except Exception as e:
            print(f"⚠️ Rerank Error: {e}")
            return candidates[0]

    # Docker / Local 경로 자동 보정 
    def _resolve_file_path(self, original_path, filename):
        # 1. DB에 적힌 원본 경로에 있으면 바로 리턴
        if original_path and os.path.exists(original_path):
            return original_path

        # 2. 도커 경로와 로컬 경로를 후보로 설정
        search_roots = [DOCKER_DATA_DIR, LOCAL_DATA_DIR]

        print(f" [Path] '{filename}' 검색 시작...")

        for root_dir in search_roots:
            if not os.path.exists(root_dir):
                continue
            
            # os.walk: 폴더를 계속 파고들면서 모든 파일을 훑는 함수
            for current_root, dirs, files in os.walk(root_dir):
                if filename in files:
                    found_path = os.path.join(current_root, filename)
                    print(f"파일 발견: {found_path}")
                    return found_path

        print(f"❌ [Path] 모든 폴더를 뒤졌지만 못찾음: {filename}")
        return None

    # Vision RAG 
    def _handle_image_query(self, query, image_path, history_text):
        try:
            img = Image.open(image_path)

            prompt = VISION_RAG_PROMPT.format(
                history=history_text, 
                query=query,          
                file_name=os.path.basename(image_path)
            )
            
            return self.llm.generate([prompt, img])
        except Exception as e:
            return f"이미지 처리 오류: {str(e)}"