FROM python:3.12-slim

RUN pip install --no-cache-dir scikit-learn

WORKDIR /app
COPY worker.py /app/worker.py

EXPOSE 8750
CMD ["python", "worker.py", "serve", "--host", "0.0.0.0", "--port", "8750"]
