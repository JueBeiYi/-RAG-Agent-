FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/models/huggingface MPLCONFIGDIR=/tmp/matplotlib
ARG INSTALL_MODELS=true
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt && if [ "$INSTALL_MODELS" = "true" ]; then pip install --no-cache-dir -r requirements-models.txt; fi
COPY src ./src
COPY scripts ./scripts
COPY config.yaml pyproject.toml ./
RUN mkdir -p /app/data/runtime /models/huggingface
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
