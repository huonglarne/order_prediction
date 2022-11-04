FROM mcr.microsoft.com/devcontainers/python:3.9

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt