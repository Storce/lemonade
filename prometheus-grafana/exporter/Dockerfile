FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install requests prometheus_client
COPY lemonade-exporter.py /app/exporter.py
WORKDIR /app
# Use the service name "lemonade" defined in docker-compose
ENTRYPOINT ["python", "exporter.py"]
CMD ["--lemonade-url", "http://lemonade:13305", "--port", "9091"]
