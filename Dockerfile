# 기본 이미지
FROM python:3.10-slim

# 패키지 설치
RUN apt-get update && apt-get install -y \
    swig \
    ffmpeg \
    xvfb \
    libgl1-mesa-glx \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 코드 복사
COPY . .

# 패키지 설치
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 학습 실행
CMD ["python", "test.py"]
