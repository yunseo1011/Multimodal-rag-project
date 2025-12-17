from paddleocr import PaddleOCR
import os
import pprint # 데이터를 예쁘게 출력해주는 도구

# 1. 모델 설정
ocr = PaddleOCR(use_textline_orientation=True, lang='korean')

# 2. 이미지 경로
img_path = 'data/raw/advertisement/doc_0007.png'

if not os.path.exists(img_path):
    print(f"오류: '{img_path}' 파일을 찾을 수 없습니다.")
    exit()

print(f"OCR 분석 시작: {img_path} ...")

# 3. OCR 실행
result = ocr.ocr(img_path)

# 4. 결과 구조 확인하기
print("데이터 구조 확인하기")

# 결과가 비어있는지 체크
if not result:
    print("❌ 결과가 없습니다 (None).")
else:
    print(f"데이터 타입: {type(result)}")
    print(f"리스트 길이: {len(result)}")
    
    # 첫 번째 데이터 확인
    print("\n첫 번째 데이터 (result[0]) 내용:")
    pprint.pprint(result[0])
