FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel AS build

RUN apt update && apt install gcc g++ git -y && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /openchat-workspace

ENV PATH=/workspace-lib:/workspace-lib/bin:$PATH
ENV PYTHONUSERBASE=/workspace-lib

COPY ochat /openchat-workspace/
COPY pyproject.toml /openchat-workspace/

RUN pip install /openchat-workspace --no-cache-dir --user

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed --no-cache-dir --user

RUN pip install flash-attn --no-build-isolation --no-cache-dir --user

RUN pip install ring_flash_attn@git+https://github.com/zhuzilin/ring-flash-attention --no-cache-dir --user

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS openchat

RUN apt update && apt install gcc g++ -y && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /openchat-workspace

COPY --from=build /workspace-lib /workspace-lib
COPY ochat/training_deepspeed/deepspeed* /openchat-workspace/config/

ENV PATH=/workspace-lib:/workspace-lib/bin:$PATH
ENV PYTHONUSERBASE=/workspace-lib
ENV PYTHONPATH=/workspace-lib:/vllm-workspace