import torch
import chromadb
import os
from PIL import Image
from src.core.model_loader import get_model

'''질문: "총액이 얼마야?"
        ↓
   _query_to_embedding() → 768차원 벡터
        ↓  
   search() → ChromaDB에서 Top-K 문서 검색
        ↓
   결과: [{"rank":1, "id":"doc_0001", "score":0.12, "label":"invoice", ...}]
'''

class SearchEngine:
    def __init__(self, db_path="./chroma_db", collection_name="docs"):
        print(f" Initializing Search Engine...")
        
        # 1. DB 연결
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        
        # 2. 모델 로드 & 디바이스 설정 (MPS 지원 추가)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # 맥북 M1/M2/M3 가속기 사용
        else:
            self.device = "cpu"
            
        self.model, self.processor = get_model()
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Search Engine Ready (Device: {self.device})")

    def _query_to_embedding(self, text_query):
        """
        텍스트 쿼리 -> 벡터 변환
        """
        # 1. Dummy Image
        dummy_image = Image.new("RGB", (224, 224), color="black")
        
        # 2. Text -> Words
        words = text_query.split()
        if not words: words = ["unknown"]
        
        # 3. Dummy BBox
        boxes = [[0, 0, 0, 0]] * len(words)

        # 4. 인코딩
        encoding = self.processor(
            images=dummy_image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        
        # 5. 추론 및 Mean Pooling
        with torch.no_grad():
            outputs = self.model(**inputs)
    
            # 모델 출력(Text + Image)에서 텍스트 길이만큼만 슬라이싱
            # inputs["input_ids"]의 길이(512)를 기준으로 자름
            seq_len = inputs["input_ids"].shape[1] 
            text_output = outputs.last_hidden_state[:, :seq_len, :] # (1, 512, 768)
            
            # 마스크 확장 및 연산
            mask = inputs["attention_mask"].unsqueeze(-1).expand(text_output.size()).float()
            
            sum_embeddings = torch.sum(text_output * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding[0].cpu().tolist()

    # ChromaDB가 "질문 벡터와 가장 비슷한 문서들"을 찾아줌
    def search(self, query, top_k=5, filter_label=None):
        query_vec = self._query_to_embedding(query)
        where_condition = {"label": filter_label} if filter_label else None
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_condition
        )
        
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "rank": i + 1,
                    "id": results['ids'][0][i],
                    "score": results['distances'][0][i],
                    "label": results['metadatas'][0][i].get('label', 'N/A'),
                    "file_path": results['metadatas'][0][i].get('file_path', 'N/A'),
                    "preview": results['documents'][0][i][:100]
                })

        return formatted_results
    