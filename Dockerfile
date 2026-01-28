# 1. 가벼운 파이썬 3.11 이미지 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치
# (build-essential, git 등을 포함해서 설치 에러 방지)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 전체 코드 복사
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 6. 서버 실행 명령
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]