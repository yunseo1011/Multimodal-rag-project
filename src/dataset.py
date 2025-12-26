import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils import normalize_box  # utils.py에서 함수 가져오기


class LayoutLMDataset(Dataset):
    def __init__(self, data_pairs, processor, label2id):
        self.data_pairs = data_pairs
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]

        # 1. Image 로드
        image = Image.open(item["image_path"]).convert("RGB")
        width, height = image.size

        # 2. JSON 로드
        with open(item["json_path"], "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        words = []
        boxes = []
        
        # 'lines' 키가 없을 수도 있으니 .get() 사용 (안전)
        for line in ocr_data.get("lines", []):
            text = line.get("text", "").strip()
            bbox = line.get("bbox", [])

            # 텍스트가 없거나 박스 좌표가 4개가 아니면 스킵
            if not text or len(bbox) != 4:
                continue

            words.append(text)
            # utils.py의 함수로 즉시 변환해서 추가 (효율적)
            boxes.append(normalize_box(bbox, width, height))


        # 3. 방어 코드 (빈 문서 처리)
        if len(words) == 0:
            words = [" "]
            boxes = [[0, 0, 0, 0]]

        # 4. Processor 호출
        encoding = self.processor(
            images=image,
            text=words, # JSON에서 온 텍스트
            boxes=boxes, # JSON에서 온 좌표
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # batch 차원 제거 (squeeze(0)이 더 안전함)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # 5. label 추가 (LongTensor 변환 필수)
        encoding["labels"] = torch.tensor(self.label2id[item["label"]], dtype=torch.long)

        return encoding