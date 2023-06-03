FROM nvcr.io/nvidia/pytorch:23.05-py3
USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN yes | apt install libpq-dev libaio-dev

COPY req.txt /opt/app/req.txt
WORKDIR /opt/app
RUN pip install -r req.txt
COPY . /opt/app

CMD uvicorn api:app --host 0.0.0.0 --port 8281
