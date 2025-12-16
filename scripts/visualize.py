import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

METADATA_FILE = "data/metadata.json"
PLOT_FILE = "dataset_distribution.png"
SAMPLE_FILE = "dataset_samples.png"

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"{METADATA_FILE} 파일이 없습니다.")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 데이터 분포 확인
    labels = [d['label_name'] for d in data]
    counts = Counter(labels)
    
    print("\n📊 [Class Distribution]")
    print(f"Total Images: {len(data)}")
    for label, count in counts.most_common():
        print(f"- {label}: {count}")

    # 2. Bar Plot 그리기
    plt.figure(figsize=(12, 6))
    sorted_counts = counts.most_common()
    x = [item[0] for item in sorted_counts]
    y = [item[1] for item in sorted_counts]

    sns.barplot(x=x, y=y, palette="viridis")
    plt.title(f"Dataset Distribution (Total: {len(data)})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"\n분포 그래프 저장 완료: {PLOT_FILE}")

    # 3. 랜덤 샘플 이미지 9장 시각화
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    samples = random.sample(data, 9)

    for i, ax in enumerate(axes.flat):
        img_info = samples[i]
        try:
            img = Image.open(img_info['image_path'])
            ax.imshow(img)
            ax.set_title(img_info['label_name'], fontsize=10)
            ax.axis('off')
        except Exception:
            ax.text(0.5, 0.5, "Error", ha='center')

    plt.tight_layout()
    plt.savefig(SAMPLE_FILE)
    print(f"샘플 이미지 저장 완료: {SAMPLE_FILE}")

if __name__ == "__main__":
    main()