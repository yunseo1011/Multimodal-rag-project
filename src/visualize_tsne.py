import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 1. 프로젝트 루트 경로 설정 (필수)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# [수정됨] 리팩토링된 모듈들 가져오기
from src.core.model_loader import get_model
from src.core.ocr_processor import process_ocr_data
from src.core.embedding import extract_embedding
from src.utils.data_utils import get_data_pairs

# 2. 설정
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 메인 실행 로직
def main():
    print("t-SNE Analysis Initializing...")

    # 모델 로딩 (Singleton)
    model, processor = get_model()
    
    # [수정됨] 데이터 수집 (utils에 있는 함수 재사용!)
    # data/raw 와 data/processed/ocr 경로를 지정
    raw_root = os.path.join(project_root, "data/raw")
    ocr_root = os.path.join(project_root, "data/processed/ocr")
    
    print("Collecting Data...")
    data_pairs = get_data_pairs(raw_root, ocr_root)
    print(f"Total documents found: {len(data_pairs)}")
    
    if len(data_pairs) < 10:
        print("데이터가 너무 적습니다. t-SNE 분석을 위해 최소 10장 이상 필요합니다.")
        return

    # 임베딩 추출
    embeddings = []
    labels = []
    
    print("\nExtracting Embeddings...")
    for item in tqdm(data_pairs):
        try:
            # [수정됨] 1. OCR 전처리
            ocr_data = process_ocr_data(item['image_path'], item['json_path'])
            
            # [수정됨] 2. 임베딩 추출
            vec = extract_embedding(model, processor, ocr_data)
            
            if vec is not None:
                embeddings.append(vec)
                labels.append(item['label'])
                
        except Exception as e:
            print(f"Error processing {item['image_path']}: {e}")
            continue
    
    # numpy 배열로 변환
    X = np.array(embeddings)
    y = np.array(labels)
    
    print(f"\nEmbedding Shape: {X.shape}") # (N, 768)

    # 4. t-SNE 차원 축소 (768D -> 2D)
    print(" Running t-SNE (Dimensionality Reduction)...")
    # perplexity는 데이터 개수보다 작아야 함 (안전장치 추가)
    perp = min(30, len(X) - 1)
    if perp < 5: perp = len(X) - 1 # 데이터가 극소량일 경우 방어

    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 5. 시각화 및 저장
    print("Plotting & Saving...")
    df = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "label": y
    })
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df, 
        x="x", y="y", 
        hue="label", 
        style="label", 
        s=100, 
        palette="viridis",
        alpha=0.8
    )
    plt.title(f"Document Embedding Clusters (t-SNE) - n={len(X)}", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "tsne_visualization.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n Analysis saved to: {save_path}")

if __name__ == "__main__":
    main()