FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./code/requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY ./code /app

EXPOSE 5010

ENV NAME botenv

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5010", "--reload"]

