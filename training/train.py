import torch
import os
import sys  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from torch.optim import AdamW  
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from core.dataset import LayoutLMDataset
from src.utils import get_data_pairs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Mac/Windows 멀티프로세싱 에러 방지용 가드
if __name__ == '__main__':
    
    # 1. Device 설정
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"Device: {DEVICE}")

    # 2. 경로 설정
    RAW_ROOT = "data/raw"
    OCR_ROOT = "data/processed/ocr"
    MODEL_ID = "microsoft/layoutlmv3-base"
    SAVE_PATH = "models/layoutlmv3_finetuned.pt"
    
    # 저장할 폴더가 없으면 미리 생성
    save_dir = os.path.dirname(SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 3. 데이터 로드 & 분할
    data_pairs = get_data_pairs(RAW_ROOT, OCR_ROOT)

    labels = sorted(list(set(d["label"] for d in data_pairs)))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    train_pairs, val_pairs = train_test_split(
        data_pairs,
        test_size=0.2,
        random_state=42,
        stratify=[d["label"] for d in data_pairs]
    )

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    print(f"Labels ({len(labels)}): {label2id}")

    # 4. Processor & Dataset
    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_ID,
        apply_ocr=False
    )

    # 이미지 정규화 설정 (NaN 방지 필수 설정)
    processor.image_processor.do_normalize = True
    processor.image_processor.image_mean = [0.5, 0.5, 0.5]
    processor.image_processor.image_std = [0.5, 0.5, 0.5]

    train_dataset = LayoutLMDataset(train_pairs, processor, label2id)
    val_dataset = LayoutLMDataset(val_pairs, processor, label2id)

    # num_workers=2 설정 시 반드시 __main__ 가드가 필요함
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2, 
        pin_memory=(DEVICE == "cuda" or DEVICE == "mps") 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=(DEVICE == "cuda" or DEVICE == "mps")
    )

    # 5. 모델 로드 (Full Fine-tuning)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    ).to(DEVICE)

    print("Strategy: Full Fine-tuning (All layers trainable)")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    EPOCHS = 10
    best_val_acc = 0.0 # 최고 점수 기록용 변수

    # 6. 학습 루프
    print("Training Started...")
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for batch in progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(DEVICE)
                batch = {
                    k: v.to(DEVICE)
                    for k, v in batch.items()
                    if k != "labels"
                }

                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Acc {val_acc:.4f} ({val_acc*100:.2f}%)"
        )

        # 7. Best Model 저장 로직
        if val_acc > best_val_acc:
            print(f"New Best Model Found! ({best_val_acc:.4f} -> {val_acc:.4f}) Saving...")
            best_val_acc = val_acc
            
            torch.save(model.state_dict(), SAVE_PATH)
            processor.save_pretrained("models/processor")

    print("Training Finished")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best Model Saved at: {SAVE_PATH}")