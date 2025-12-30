import chromadb
import pandas as pd
import os
from tqdm import tqdm

# 설정 
DB_PATH = "./chroma_db"
DATA_PATH = "data/processed/document_embeddings.parquet"
COLLECTION_NAME = "docs"
BATCH_SIZE = 100 

def main():
    # 1. DB 연결 (Persistent: 디스크에 저장)
    print(f" Connecting to ChromaDB at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Collection 생성 (Cosine Similarity)
    collection = client.get_or_create_collection( # 있으면 가져오고, 없으면 새 컬렉션 생성
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Collection '{COLLECTION_NAME}' ready.")

    # 3. 데이터 로드
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f" 파일을 찾을 수 없습니다: {DATA_PATH}")
    
    print(f" Reading Parquet from '{DATA_PATH}'...")
    df = pd.read_parquet(DATA_PATH)
    total_docs = len(df)
    
    print(f"📊 Total documents: {total_docs}")
    print(f"   Columns: {df.columns.tolist()}")

    # 4. 데이터 준비 (깔끔해진 매핑)
    ids = df["doc_id"].astype(str).tolist()
    embeddings = df["embedding"].tolist()
    documents = df["text"].fillna("").tolist() # 텍스트 컬럼
    
    # 메타데이터 생성 (Parquet의 label 컬럼을 바로 사용)
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "label": str(row["label"]),      # 라벨 (문자열로 저장 추천)
            "file_path": str(row["file_path"])
        })

    # 5. DB 적재 (Upsert)
    print(" Starting ingestion...")
    
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Ingesting"):
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

    # 6. 최종 검증
    final_count = collection.count()
    print(f"\n🎉 Ingestion Complete!")
    print(f"📉 Total Documents in DB: {final_count}")
    
    if final_count > 0:
        # 데이터 하나만 살짝 꺼내서 라벨 잘 들어갔나 확인
        sample = collection.peek(1)
        print("\n🔍 Sample Check:")
        print(f" - ID: {sample['ids'][0]}")
        print(f" - Metadata: {sample['metadatas'][0]}")
        print(" SUCCESS: DB 적재 및 라벨 저장 완료!")

if __name__ == "__main__":
    main()