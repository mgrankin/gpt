FROM nvcr.io/nvidia/pytorch:23.08-py3
USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN yes | apt install libpq-dev libaio-dev
