# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy model code and scripts
COPY model_structure/ /app/model_structure/
COPY model_weights/ /app/model_weights/
COPY pretrain_model_with_ATAC.py fine_tune_model.py predict.py attribution_utils.py /app/

# Install required Python packages
# Captum is not included in base PyTorch images, so install it explicitly.
RUN pip install --no-cache-dir \
    captum \
    tqdm \
    pandas \
    numpy \
    scikit-learn \
    h5py

# Create and switch to a non-root user for safety
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command (run a Python script)
ENTRYPOINT ["python"]
