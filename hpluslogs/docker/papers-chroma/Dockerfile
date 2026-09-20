FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=False TIKTOKEN_CACHE_DIR=/app/tiktoken-cache
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
COPY hpluslogs /app/hpluslogs
ENTRYPOINT ["python", "-m", "hpluslogs.papers_chroma_cli"]
