# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY model_structure/ /app/model_structure/
COPY encode_fasta.py /app/
COPY predict.py /app/
COPY run_prediction.py /app/

# Copy pretrained model weights (from relative path)
COPY model_weights/best_model_dELSs.pth /app/model_weights/

# Default entrypoint
ENTRYPOINT ["python", "run_prediction.py"]

