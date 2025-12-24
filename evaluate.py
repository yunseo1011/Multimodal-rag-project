import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import LayoutLMDataset
from utils import get_data_pairs

# ==========================================
# 1. 설정
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

RAW_ROOT = "data/raw"
OCR_ROOT = "data/processed/ocr"
MODEL_PATH = "models/layoutlmv3_finetuned.pt"
BASE_MODEL = "microsoft/layoutlmv3-base"

def evaluate():
    print(f"🕵️‍♂️ Evaluation Started on {DEVICE}...")

    # 데이터 로드
    data_pairs = get_data_pairs(RAW_ROOT, OCR_ROOT)
    labels = sorted(list(set(d["label"] for d in data_pairs)))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    _, val_pairs = train_test_split(
        data_pairs, test_size=0.2, random_state=42, stratify=[d["label"] for d in data_pairs]
    )
    
    print(f"📊 Validation Samples: {len(val_pairs)}")
    print(f"🏷️ Labels: {label2id}")

    # Processor
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)
    processor.image_processor.do_normalize = True
    processor.image_processor.image_mean = [0.5, 0.5, 0.5]
    processor.image_processor.image_std = [0.5, 0.5, 0.5]

    dataset = LayoutLMDataset(val_pairs, processor, label2id)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 모델 로드
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), label2id=label2id, id2label=id2label
    )
    
    print(f"💾 Loading model weights from {MODEL_PATH}...")
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(MODEL_PATH)
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 추론
    print("🚀 Running Inference...")
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            pred_labels.extend(preds.cpu().numpy())

    true_labels = [label2id[p['label']] for p in val_pairs]

    # 결과 출력
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n🏆 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n📜 Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=labels))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("📈 Confusion Matrix saved as 'confusion_matrix.png'")

    # ==========================================
    # ✅ 정답 맞춘 케이스 확인 (Success Samples)
    # ==========================================
    print("\n========================================")
    print("✅ Success Samples (Good Examples)")
    print("========================================")

    success_count = 0
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true == pred:
            success_count += 1
            info = val_pairs[i]
            print(f"✨ Success Case #{success_count}")
            print(f"   📂 File: {info['image_path']}")
            print(f"   🎯 Label: {id2label[true]}")
            
            try:
                with open(info['json_path'], 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # [수정된 로직 적용]
                if 'words' in ocr_data:
                    words = ocr_data['words']
                elif 'full_text' in ocr_data:
                    words = ocr_data['full_text'].split()
                else:
                    words = []

                if len(words) > 0:
                    text_snippet = " ".join(words[:15]) + "..." if len(words) > 15 else " ".join(words)
                else:
                    text_snippet = "(Empty - No text detected)"
                    
                print(f"   📝 OCR snippet: {text_snippet}")
                
            except Exception as e:
                print(f"   ⚠️ OCR Load Failed: {e}")

            print("-" * 40)
            if success_count >= 5: break

    # ==========================================
    # 💀 오답 노트 (Failed Samples)
    # ==========================================
    print("\n========================================")
    print("💀 Failed Samples (Why did it fail?)")
    print("========================================")
    
    wrong_count = 0
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            wrong_count += 1
            info = val_pairs[i]
            print(f"❌ Case #{wrong_count}")
            print(f"   📂 File: {info['image_path']}")
            print(f"   ✅ Truth: {id2label[true]}  |  🤖 Pred: {id2label[pred]}")
            
            try:
                with open(info['json_path'], 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # [수정된 로직 적용]
                if 'words' in ocr_data:
                    words = ocr_data['words']
                elif 'full_text' in ocr_data:
                    words = ocr_data['full_text'].split()
                else:
                    words = []
                    
                text_snippet = " ".join(words[:15]) + "..." if len(words) > 15 else " ".join(words)
                print(f"   📝 OCR snippet: {text_snippet}")
                
            except Exception as e:
                print(f"   ⚠️ OCR Load Failed: {e}")

            print("-" * 40)
            if wrong_count >= 10: break

if __name__ == "__main__":
    evaluate()