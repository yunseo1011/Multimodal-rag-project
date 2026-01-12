# 1. 가벼운 파이썬 3.10 이미지 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (CV2나 기타 라이브러리 대비)
# apt-get update 후 설치하고, 캐시를 지워서 용량을 줄임
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1\
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
# (코드를 복사하기 전에 먼저 함으로써 캐싱 효과를 냄 -> 빌드 속도 UP)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 전체 코드 복사
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 6. 서버 실행 명령
# host를 0.0.0.0으로 해야 외부에서 접속 가능
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]