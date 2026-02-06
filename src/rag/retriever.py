# src/rag/retriever.py

import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self):
        # 1. DB 경로 설정
        self.db_path = "./chroma_db"
        self.collection_name = "docs"
        
        # 2. 임베딩 함수 설정 
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 없습니다.")

        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/gemini-embedding-001",  # 모델 명시
            task_type="RETRIEVAL_QUERY" # 질문할 때는 QUERY 타입 사용
        )

        # 3. DB 연결
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 4. 컬렉션 가져오기
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✅ Retriever Connected to ChromaDB at '{self.db_path}'")
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            raise e

    def retrieve(self, query: str, top_k: int = 5, category: str = None):
        """
        질문을 받아서 관련된 문서를 찾아옵니다.
        """
        try:
            # 검색 필터 (Router가 카테고리를 줬으면 그걸로 필터링)
            where_filter = None
            if category:
                where_filter = {"label": category}

            # 검색 실행
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter 
            )
            
            # 보기 좋게 정리해서 반환
            docs = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    docs.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })
            
            return docs

        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생: {e}")
            return []