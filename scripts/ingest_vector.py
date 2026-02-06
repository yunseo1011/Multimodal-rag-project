# scripts/ingest_vector.py 
import chromadb
import chromadb.utils.embedding_functions as embedding_functions # 추가됨
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# 설정 
DB_PATH = "./chroma_db"
DATA_PATH = "data/processed/document_embeddings.parquet"
COLLECTION_NAME = "docs"
BATCH_SIZE = 100 

def main():
    # 1. DB 연결
    print(f" Connecting to ChromaDB at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Gemini 번역기 설정 
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="RETRIEVAL_QUERY"
    )

    # 2. 기존에 잘못 만들어진 DB가 있다면 삭제 
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f" 기존 '{COLLECTION_NAME}' 컬렉션 삭제 완료 (초기화)")
    except:
        pass # 없으면 넘어감

    # 3. 컬렉션 다시 생성 
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=gemini_ef, # 추가
        metadata={"hnsw:space": "cosine"}
    )
    print(f" Collection '{COLLECTION_NAME}' created (with Gemini Config).")

    # 4. 데이터 로드 
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f" 파일을 찾을 수 없습니다: {DATA_PATH}")
    
    print(f"Reading Parquet from '{DATA_PATH}'...")
    df = pd.read_parquet(DATA_PATH)
    
    ids = df["doc_id"].astype(str).tolist()
    embeddings = [x.tolist() for x in df["embedding"]]
    documents = df["text"].fillna("").tolist() 
    
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "label": str(row["label"]),      
            "file_path": str(row["file_path"])
        })

    # 5. DB 적재
    print("Starting ingestion...")
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Ingesting"):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        batch_documents = documents[i : i + BATCH_SIZE]
        batch_metadatas = metadatas[i : i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )


if __name__ == "__main__":
    main()