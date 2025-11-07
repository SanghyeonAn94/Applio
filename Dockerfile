# syntax=docker/dockerfile:1
FROM python:3.10-bullseye

# Expose the required port
EXPOSE 6969

# Install system dependencies, clean up cache to keep image size small
RUN apt update && \
    apt install -y -qq ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create python symlink for consistency with GPT-SoVITS
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /workspace

# Create Applio directory (will be mounted via volumes)
RUN mkdir -p /workspace/Applio

# Set working directory to Applio
WORKDIR /workspace/Applio

# Copy requirements.txt for dependency installation
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA 12.8 support first
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install python-ffmpeg
RUN pip install --no-cache-dir python-ffmpeg

# Install Applio dependencies from requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Run the API server (no ENTRYPOINT, command specified in docker-compose)
# CMD ["python", "api.py", "-a", "0.0.0.0", "-p", "6969"]
