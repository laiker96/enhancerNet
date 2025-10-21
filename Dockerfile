# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy project files
COPY model_structure/ /app/model_structure/
COPY model_weights/ /app/model_weights/
COPY pretrain_model_with_ATAC.py train_model_dELSs.py encode_fasta.py predict.py run_prediction.py /app/

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Set default command (can be overridden)
CMD ["python", "run_prediction.py"]