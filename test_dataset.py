import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor
from utils import get_data_pairs
from dataset import LayoutLMDataset

# 1. 경로 및 모델 설정
RAW_ROOT = "data/raw"              # 원본 이미지 폴더
OCR_ROOT = "data/processed/ocr"    # OCR 결과(JSON) 폴더
MODEL_ID = "microsoft/layoutlmv3-base"


def test_dataset():
    print("데이터셋 통합 테스트 시작...")

    # 1단계: 데이터 및 프로세서 로드
    
    # 1. 데이터 짝(Pair) 가져오기 (JSON 없는 파일 자동 Skip 확인)
    data_pairs = get_data_pairs(RAW_ROOT, OCR_ROOT)
    if not data_pairs:
        print("데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 라벨 맵(Label Map) 만들기
    labels = sorted(list(set(p['label'] for p in data_pairs)))
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"감지된 라벨 ({len(label2id)}개): {label2id}")

    # 3. Processor 로드 (apply_ocr=False 필수)
    print("Processor 로드 중...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)

    # 4. Dataset 생성
    dataset = LayoutLMDataset(data_pairs, processor, label2id)
    print(f"Dataset 생성 완료! 총 데이터 개수: {len(dataset)}개")

    # 2단계: 단일 샘플 정밀 검사 (__getitem__)
    print("\n[Test 1] 단일 샘플 데이터 구조 확인 중...")
    sample = dataset[0]

    # 주요 키 존재 여부 확인
    required_keys = ['input_ids', 'attention_mask', 'bbox', 'pixel_values', 'labels']
    for key in required_keys:
        if key not in sample:
            print(f"실패: '{key}' 키가 데이터에 없습니다.")
            return

    # 차원(Shape) 검증
    print(f" - Image Shape: {sample['pixel_values'].shape}  (Target: [3, 224, 224])")
    print(f" - Input IDs:   {sample['input_ids'].shape}      (Target: [512])")
    print(f" - BBox Shape:  {sample['bbox'].shape}           (Target: [512, 4])")
    print(f" - Label:       {sample['labels']}               (Type: {type(sample['labels'])})")

    # Assert 검증
    assert sample['pixel_values'].shape == (3, 224, 224), "이미지 크기 오류"
    assert sample['input_ids'].shape == (512,), "토큰 길이 오류"
    assert sample['bbox'].shape == (512, 4), "박스 좌표 오류"
    assert isinstance(sample['labels'], torch.LongTensor) or isinstance(sample['labels'], torch.Tensor), "라벨 Tensor 변환 안됨"
    print("단일 샘플 검증 통과!")

    # 3단계: DataLoader 배치 처리 테스트 (Batch)
    print("\n[Test 2] DataLoader 배치(Batch) 테스트 중...")
    
    BATCH_SIZE = 2
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    try:
        # 배치 하나만 뽑아보기
        batch = next(iter(dataloader))
        
        # 배치 차원이 잘 붙었는지 확인 (Batch Size, ...)
        print(f" - Batch Input IDs: {batch['input_ids'].shape}   (Target: [{BATCH_SIZE}, 512])")
        print(f" - Batch Images:    {batch['pixel_values'].shape} (Target: [{BATCH_SIZE}, 3, 224, 224])")
        print(f" - Batch Labels:    {batch['labels'].shape}       (Target: [{BATCH_SIZE}])")

        # Assert 검증
        assert batch['input_ids'].shape == (BATCH_SIZE, 512), "배치 Input IDs 차원 오류"
        assert batch['pixel_values'].shape == (BATCH_SIZE, 3, 224, 224), "배치 이미지 차원 오류"
        assert batch['bbox'].shape == (BATCH_SIZE, 512, 4), "배치 BBox 차원 오류"
        assert batch['labels'].shape == (BATCH_SIZE,), "배치 라벨 차원 오류"
        
        print("배치 처리 검증 통과!")
        
    except Exception as e:
        print(f"DataLoader 테스트 실패: {e}")
        return

    print("\nDataset과 DataLoader가 완벽하게 동작합니다.")

if __name__ == "__main__":
    test_dataset()