FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/models/huggingface MPLCONFIGDIR=/tmp/matplotlib
ARG INSTALL_MODELS=true
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Install the CPU wheel explicitly: PyPI's Linux torch wheels can otherwise pull
# several GB of CUDA libraries although embedding_device is configured as CPU.
RUN if [ "$INSTALL_MODELS" = "true" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3' && \
      pip install --no-cache-dir -r requirements-models.txt; \
    fi
COPY src ./src
COPY scripts ./scripts
COPY config.yaml config.standard.yaml pyproject.toml ./
RUN mkdir -p /app/data/runtime /models/huggingface/hub
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
