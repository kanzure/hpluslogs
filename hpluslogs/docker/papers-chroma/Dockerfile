FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=False HOME=/opt/chroma-model
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['Cache the local model.'])"
COPY hpluslogs /app/hpluslogs
ENTRYPOINT ["python", "-m", "hpluslogs.papers_chroma_cli"]
