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

# Copy requirements.txt for dependency installation
COPY requirements.txt /tmp/requirements.txt

# Install typing-extensions first from PyPI to avoid naming conflict
RUN pip install --no-cache-dir typing-extensions>=4.10.0

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install python-ffmpeg
RUN pip install --no-cache-dir python-ffmpeg

# Install Applio dependencies from requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace/Applio

# Copy Applio source code
COPY . /workspace/Applio

# Backup mute directories (will be restored if volume mount is empty)
RUN cp -r /workspace/Applio/logs/mute* /tmp/ 2>/dev/null || true

# Create entrypoint script to restore mute files if needed
RUN echo '#!/bin/bash\n\
# Restore mute directories if they do not exist in mounted volume\n\
if [ ! -d "/workspace/Applio/logs/mute" ]; then\n\
  echo "Initializing mute directories..."\n\
  cp -r /tmp/mute* /workspace/Applio/logs/ 2>/dev/null || true\n\
  echo "Mute directories initialized"\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
