FROM mcr.microsoft.com/devcontainers/python:3.9 AS base

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

FROM base AS production

WORKDIR /app
COPY . /app

CMD ["uvicorn", "src.prediction_api:app", "--host", "0.0.0.0", "--port", "80"]