FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2

# libgl1/libglib are needed by the OpenCV build that RapidOCR pulls in
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY main.py .

RUN useradd -m bot && chown -R bot /app
USER bot
CMD ["python", "main.py"]
