FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba \
    HEATSIM_BACKEND=numba \
    HEATSIM_RESULTS_DIR=/data/jobs

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY heatsim/ heatsim/
COPY service/ service/

RUN useradd --create-home app && mkdir -p /data/jobs && chown -R app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
