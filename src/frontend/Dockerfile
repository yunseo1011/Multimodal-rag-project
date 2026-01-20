FROM python:3.11-slim

WORKDIR /app

COPY src/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 포트 개방 (Streamlit 기본 포트)
EXPOSE 8501

# (0.0.0.0으로 열어야 도커 밖에서 접속 가능)
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]