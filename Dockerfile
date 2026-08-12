# Runs the Streamlit application against a pre-trained model.
#
# The image deliberately does not collect data or train: collection takes over an
# hour against a rate-limited public API, so models/model.joblib is built once on
# the host with `python -m src.train` and copied in. That keeps `docker compose up`
# from a clean clone to a few seconds.
FROM python:3.13-slim

WORKDIR /app

# Dependencies first, so editing source does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src/ holds everything the app depends on; app/ is only the interface.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/raw/block_anchors.json ./data/raw/block_anchors.json
COPY reports/metrics.csv ./reports/metrics.csv

EXPOSE 8501

# Without this Streamlit's own health check reports the container as unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
