FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY docs /app/docs
COPY run_app.py /app/run_app.py

EXPOSE 8011

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8011"]
