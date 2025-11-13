# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
WORKDIR /app

COPY model_structure/ /app/model_structure/
COPY model_weights/ /app/model_weights/
COPY pretrain_model_with_ATAC.py fine_tune_model.py predict.py /app/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python"]
