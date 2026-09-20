FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PAPERS_INFERENCE_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY hpluslogs /app/hpluslogs
ENTRYPOINT ["python", "-m", "hpluslogs.conversion_cli"]
