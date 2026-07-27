# syntax=docker/dockerfile:1
FROM python:3.12-bookworm

EXPOSE 6969

RUN apt update && \
    apt install -y -qq ffmpeg libportaudio2 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir typing-extensions>=4.10.0

RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir python-ffmpeg

RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/Applio

COPY . /workspace/Applio

RUN cp -r /workspace/Applio/logs/mute* /tmp/ 2>/dev/null || true

RUN echo '#!/bin/bash\n\
# Restore mute directories if they do not exist in mounted volume\n\
if [ ! -d "/workspace/Applio/logs/mute" ]; then\n\
  echo "Initializing mute directories..."\n\
  cp -r /tmp/mute* /workspace/Applio/logs/ 2>/dev/null || true\n\
  echo "Mute directories initialized"\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
