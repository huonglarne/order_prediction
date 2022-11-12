FROM nvcr.io/nvidia/tritonserver:22.09-py3 AS base

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

FROM base AS production

COPY ./models /models

EXPOSE 8001

CMD ["tritonserver", "--model-repository=/models", "--http-port=8001", "--allow-metrics=false", "--allow-http=true", "--exit-timeout-secs=3", "--log-verbose=3", "--strict-model-config=false"]
