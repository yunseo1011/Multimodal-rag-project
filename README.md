# 📂 Multimodal RAG Project

문서 이미지(Document Image)와 텍스트 정보를 함께 이해하여  
**Retrieval-Augmented Generation(RAG)** 시스템을 구축하는 멀티모달 프로젝트입니다.

본 프로젝트는  
**문서 이미지 수집 → 전처리 → 문서 이해 모델 → 검색 → 생성**으로 이어지는  
End-to-End 파이프라인 구축을 목표로 합니다.

현재는 **Phase 1 (인프라 + 데이터 & OCR 파이프라인)**을 완료했고,  
다음 단계인 멀티모달 모델 학습 및 RAG 시스템 구현으로 넘어갈 준비가 된 상태입니다.

---

## Project Status

- [x] **Phase 1: 인프라 구축 및 데이터 파이프라인**
  - RVL-CDIP 기반 문서 이미지 Subset 구축 및 메타데이터 생성
  - RAG 전용 OCR 파이프라인(텍스트+좌표 JSON) 완성
  - OCR HTTP API (FastAPI) 및 문서 단위 JSON 저장 기능 구현
- [ ] **Phase 2: 멀티모달 모델 학습 및 임베딩**
- [ ] **Phase 3: RAG 시스템 구현 및 컨테이너화**
- [ ] **Phase 4: 클라우드 배포 및 문서화**

---

## 📊 Dataset

HuggingFace의 `rvl_cdip` 데이터셋을 **Streaming 방식**으로 수집하여  
문서 이미지 분류 및 멀티모달 학습에 활용 가능한 Subset을 구축했습니다.

- **Total Images:** 1,000
- **Classes:** 16  
  (Scientific publication, Budget, Invoice, Resume 등)
- **Dataset Processing**
  - 클래스 균형을 고려한 Subset 구성
  - 깨진 이미지(Corrupt Image) 전수 검사 완료
  - 학습 및 서빙 공용 메타데이터 생성 (`data/metadata.json`)
- **Format:**
  - Raw: `.png`
  - Processed: `.json` (Metadata + Full Text + BBox Info)

### Dataset Distribution
![Data Distribution](assets/dataset_distribution.png)

### Dataset Samples
![Data Samples](assets/dataset_samples.png)

---

## Tech Stack

- **Language:** Python 3.10+
- **OCR Engine:** PaddlePaddle, PaddleOCR
- **Backend:** FastAPI, Uvicorn
- **Data Processing:** Pydantic (Schema), Pillow, NumPy
- **Dataset:** HuggingFace Datasets
---


### 주요 처리
- confidence threshold 적용 (`< 0.5` 제거)
- bbox sanity check + 이미지 범위 내 clamp
- 좌상단 → 우하 방향 읽기 순서 정렬
- 문서 단위 JSON 저장 (`data/processed/ocr/class별/*.json`)

### OCR API
- **POST /ocr**: 이미지 업로드 → RAG용 OCR JSON 반환
- **GET /**: Health check

---

## ✅ Phase 1 Summary

- RVL-CDIP Subset 1,000장 구축 완료
- PaddleOCR 기반 RAG 전용 OCR 파이프라인 완성
- FastAPI 마이크로서비스 구현 (`POST /ocr`)
- RAG 청킹/임베딩에 바로 사용 가능한 JSON 스키마 확정

## Next Step: Phase 2. Multimodal Model & Embedding

OCR로 추출된 텍스트와 문서 이미지를 결합하여, **문서의 의미를 이해하고 검색(Retrieval) 가능한 벡터로 변환**하는 단계입니다.

### 1. Multimodal Classification Model
- **Modeling:** `microsoft/layoutlmv3-base` 사전 학습 모델 도입
- **Training Strategy:** Encoder Freeze 및 Classification Head 학습 (Linear Probing)
- **Goal:** Validation Accuracy **85% 이상** 달성 및 Confusion Matrix를 통한 클래스 불균형 분석
- **Process:** HuggingFace Processor를 활용하여 이미지-텍스트 멀티모달 입력 파이프라인 구축

### 2. Embedding System for RAG
- **Vectorization:** 학습된 모델의 `[CLS]` 토큰을 활용한 **Dense Vector (768-dim)** 추출기 구현
- **Visualization:** t-SNE를 활용하여 문서 임베딩 공간의 군집화(Clustering) 성능 검증
- **Output:** 문서별 임베딩 벡터 데이터베이스(Pickle/Parquet) 구축

### 3. Engineering & Refactoring
- **Code Modularization:** 실험용 Notebook(`ipynb`) 코드를 배포 가능한 모듈(`train.py`, `dataset.py`, `evaluate.py`)로 분리
- **API Scalability:** FastAPI `APIRouter`를 도입하여 엔드포인트 관리 효율화 및 모델 서빙 구조 개선