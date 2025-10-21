# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy your model structure and scripts
COPY model_structure/ /app/model_structure/
COPY encode_fasta.py /app/
COPY predict.py /app/
COPY run_prediction.py /app/

# Default entrypoint
ENTRYPOINT ["python", "run_prediction.py"]

