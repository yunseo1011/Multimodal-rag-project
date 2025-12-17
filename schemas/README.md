# RAG용 OCR 데이터 스키마 (OCR Output Schema)

이 문서는 멀티모달 RAG 시스템에서 OCR 엔진의 출력값을 표준화하기 위한 스키마 설명서입니다.

## 설계 핵심 전략
1. **RAG Chunking 최적화**: LLM에게 문맥을 제공하기 위한 `full_text` 필드를 최상단에 배치.
2. **좌표계 (Coordinates)**: **절대 픽셀 좌표 (Absolute Pixel Coordinates)** 사용.
    - **이유**: 검색 결과 원본 이미지 하이라이팅(Highlighting) 및 크롭(Crop) 시 연산 비용 최소화.
    - **주의**: LayoutLM 학습 등 정규화(0~1000) 좌표가 필요한 경우, `metadata`의 `width`, `height`를 이용해 변환하여 사용.

## 주요 필드 상세

### 1. `full_text` (String)
- **용도**: 텍스트 임베딩(Embedding) 및 LLM Context 주입용.
- **형식**: 모든 `lines`의 텍스트를 `\n`으로 이어붙인 문자열.

### 2. `lines` (Array)
- **용도**: 사용자가 검색한 키워드가 이미지의 **어디에 있는지** 찾기 위함.
- **구조**:
    ```json
    {
      "text": "Invoice",
      "bbox": [100, 200, 300, 250],  // [xmin, ymin, xmax, ymax] (px)
      "confidence": 0.98
    }
    ```

### 3. `metadata` (Object)
- **용도**: 좌표 해석의 기준점. 이미지가 리사이징되더라도 원본 비율을 역추적하기 위해 필요.
- **필수값**: `file_name`, `image_width`, `image_height`

## 활용 예시 (Workflow)
1. **OCR 실행**: 이미지 -> PaddleOCR -> `ocr_schema.json` 포맷의 JSON 생성
2. **DB 저장**: 
    NoSQL (MongoDB):
    └── doc_id: "doc_0007", page_id: 1 → 전체 JSON 저장

    Vector DB (Pinecone/Chroma):
    └── 청크1: "INSERTION ORDER\nCUNNINGHAM..." 
        └── metadata: {"doc_id": "doc_0007", "page_id": 1,"line_range": [0, 5]}
    └── 청크2: "37S N MICHIGAN AVE\nDATE: 03/20/04..."
        └── metadata: {"doc_id": "doc_0007", "page_id": 1, "line_range": [6, 10]}
        
3. **검색(Retrieval)**:
    ① Vector DB에서 청크 검색 → "INSERTION ORDER" 관련 청크 Top-K 획득
    ② 청크 metadata로 NoSQL 조회 → 해당 page_id의 원본 JSON 획득
    ③ lines[line_range[0]:line_range[1]]로 정확한 bbox 위치 확인
    ④ 이미지 하이라이팅: bbox 좌표로 사각형 그리기
