FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY rag.py .
COPY Test.pdf .

RUN mkdir -p /app/chroma_db

CMD ["python", "rag.py"]