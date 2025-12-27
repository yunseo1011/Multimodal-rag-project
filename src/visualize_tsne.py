import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding import get_embedding_model, extract_embedding

# 1. 설정
DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")          # 이미지 폴더
OCR_DIR = os.path.join(DATA_ROOT, "processed/ocr") # JSON 폴더
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 데이터 수집 함수
def collect_data_pairs():
    pairs = []
    # raw 폴더 안의 모든 클래스(폴더) 탐색
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    for label in classes:
        # 해당 클래스의 이미지 파일들 찾기
        image_files = glob.glob(os.path.join(RAW_DIR, label, "*.*"))
        
        for img_path in image_files:
            if not img_path.lower().endswith(('.png')):
                continue
                
            file_name = os.path.basename(img_path)
            # 매칭되는 JSON 파일 찾기 (확장자만 json으로 변경)
            json_name = os.path.splitext(file_name)[0] + ".json"
            json_path = os.path.join(OCR_DIR, label, json_name)
            
            # 이미지와 JSON 둘 다 있을 때만 리스트에 추가
            if os.path.exists(json_path):
                pairs.append({
                    "image": img_path,
                    "json": json_path,
                    "label": label,
                    "filename": file_name
                })
    
    return pairs

# 3. 메인 실행 로직
def main():
    # 모델 로딩
    model, processor = get_embedding_model()
    
    # 데이터 쌍 찾기
    data_pairs = collect_data_pairs()
    print(f"Total documents found: {len(data_pairs)}")
    
    if len(data_pairs) < 10:
        print("데이터가 너무 적습니다. 최소 10장 이상 필요합니다.")
        return

    # 임베딩 추출 (Batch Processing)
    embeddings = []
    labels = []
    
    print("\nExtracting Embeddings...")
    for item in tqdm(data_pairs):
        # Mean Pooling 전략
        vec = extract_embedding(model, processor, item['image'], item['json'], strategy="mean")
        
        if vec is not None:
            embeddings.append(vec)
            labels.append(item['label'])
    
    # numpy 배열로 변환
    X = np.array(embeddings)
    y = np.array(labels)
    
    print(f"\n✅ Embedding Shape: {X.shape}") # (N, 768)

    # 4. t-SNE 차원 축소 (768D -> 2D)
    print("Running t-SNE (Dimensionality Reduction)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_embedded = tsne.fit_transform(X)
    
    # 5. 시각화 및 저장
    print("Plotting...")
    df = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "label": y
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x="x", y="y", 
        hue="label", 
        style="label", 
        s=100, 
        palette="viridis",
        alpha=0.8
    )
    plt.title("Document Embedding Clusters (t-SNE)", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "tsne_visualization.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n Analysis saved to: {save_path}")
    print(" Open the image to check clusters!")

if __name__ == "__main__":
    main()