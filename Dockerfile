FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN apt update && apt install gcc g++ git tmux htop -y && apt clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/workspace-lib:/workspace-lib/bin:$PATH
ENV PYTHONUSERBASE=/workspace-lib
ENV PYTHONPATH=/workspace-lib:/openchat-workspace

WORKDIR /openchat

COPY . /openchat/

RUN pip install . --no-cache-dir --user

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed --no-cache-dir --user

RUN pip install flash-attn --no-build-isolation --no-cache-dir --user

RUN pip install ring_flash_attn@git+https://github.com/zhuzilin/ring-flash-attention --no-cache-dir --user

WORKDIR /openchat-workspace

RUN rm -rf /openchat

COPY ochat/training_deepspeed/deepspeed* /openchat-workspace/config/