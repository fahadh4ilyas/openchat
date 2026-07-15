FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

RUN apt update && apt install gcc g++ git tmux htop libaio-dev libcublas-12-1 libcublas-dev-12-1 -y && apt clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/workspace-lib:/workspace-lib/bin:$PATH
ENV PYTHONUSERBASE=/workspace-lib
ENV PYTHONPATH=/workspace-lib:/openchat-workspace

WORKDIR /

RUN git clone https://github.com/NVIDIA/cutlass.git

ENV CUTLASS_PATH=/cutlass

WORKDIR /openchat

COPY . /openchat/

RUN pip install . --no-cache-dir --user

RUN pip install deepspeed --no-cache-dir --user

RUN pip install flash-attn --no-build-isolation --no-cache-dir --user

RUN pip install ring_flash_attn@git+https://github.com/zhuzilin/ring-flash-attention --no-cache-dir --user

WORKDIR /openchat-workspace

RUN rm -rf /openchat

COPY ochat/deepspeed_config/deepspeed* /openchat-workspace/config/