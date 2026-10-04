# Python той же версии, что при обучении.
FROM python:3.12-slim
WORKDIR /app
# Сначала зависимости для кеширования слоя.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
# Сервис запускается без root.
RUN useradd --create-home appuser
USER appuser
ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
