# FROM mcr.microsoft.com/devcontainers/python:3.9 AS base

# COPY requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt

# FROM base AS production

# WORKDIR /app
# COPY . /app

# CMD ["uvicorn", "src.prediction_api:app", "--host", "0.0.0.0", "--port", "80"]


FROM nvcr.io/nvidia/tritonserver:22.09-py3
COPY ./models /models
CMD ["tritonserver", "--model-repository=/models", "--grpc-port=8001", "--allow-metrics=false", "--allow-http=false", "--exit-timeout-secs=300", "--log-verbose=3", "--strict-model-config=false"]
