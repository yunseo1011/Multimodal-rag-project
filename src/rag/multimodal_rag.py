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
        self.text_rag = TextRAG()

    def answer(self, query: str, category: str = None, history: list = None, fixed_file_path: str = None):
        # fixed_file_path가 있으면 검색을 건너뛰고 해당 파일을 사용
        file_path = fixed_file_path
        top_doc = None

        # 검색용 쿼리: 최근 대화 일부를 섞어 의미 보강
        search_query = query
        if history:
            search_query = " ".join(history[-3:]) + " " + query

        if not file_path:
            # 문서 검색
            retrieved_docs = self.retriever.retrieve(search_query, top_k=5, category=category)
            if not retrieved_docs:
                return "검색 결과가 없습니다.", None

            # 가장 질문에 잘 맞는 문서 선택
            top_doc = self._select_best_doc(search_query, retrieved_docs)

            original_path = top_doc["metadata"].get("file_path", "")
            filename = top_doc["metadata"].get("filename", os.path.basename(original_path))

            # 실제 파일 경로 찾기
            file_path = self._resolve_file_path(original_path, filename)
            if not file_path:
                return "파일을 찾을 수 없습니다.", None

        else:
            # 이미 선택된 파일이 있으면 그 파일만 사용
            filename = os.path.basename(file_path)
            content = ""

            # 텍스트 파일이면 내용을 읽어서 TextRAG에 전달
            if not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()[:5000]
                except:
                    pass

            top_doc = {
                "content": content,
                "metadata": {"filename": filename}
            }

        # Vision 쪽에서만 사용할 대화 기록 요약
        history_text = ""
        if history:
            history_text = " ".join(history[-5:]) + " "

        # 파일 타입에 따라 Vision / Text 분기
        ext = os.path.splitext(file_path)[1].lower()

        if ext in (".png", ".jpg", ".jpeg"):
            response = self._handle_image_query(query, file_path, history_text)
        else:
            augmented_query = f"{history_text}{query}"
            response = self.text_rag.answer(augmented_query, [top_doc])

        # 답변과 함께 사용한 파일 경로 반환 (세션에 고정시키기 위함)
        return response, file_path


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