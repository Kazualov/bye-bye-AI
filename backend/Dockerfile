FROM python:3.12-slim

# System libraries required by opencv-python
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Docker containers have no GPU access in our setup (and none at all on macOS),
# so install the CPU-only PyTorch wheels first. This keeps the image several GB
# smaller than the default CUDA build. The rest of requirements.txt then sees
# torch/torchvision as already satisfied.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
