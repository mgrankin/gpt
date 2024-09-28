FROM nvcr.io/nvidia/pytorch:23.08-py3
#FROM nvcr.io/nvidia/pytorch:24.09-py3
USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN yes | apt install libpq-dev libaio-dev

COPY req.txt /opt/app/req.txt
WORKDIR /opt/app
RUN --mount=type=cache,target=/root/.cache/pip pip install -r req.txt
COPY . /opt/app

# bbb
CMD uvicorn api:app --host 0.0.0.0 --port 8200
# aaa
#CMD uvicorn api:app --host 0.0.0.0 --port 8281
