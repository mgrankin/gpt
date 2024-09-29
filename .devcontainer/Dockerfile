FROM nvcr.io/nvidia/pytorch:24.09-py3
#FROM vllm/vllm-openai:v0.6.2
USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN yes | apt install libpq-dev libaio-dev 