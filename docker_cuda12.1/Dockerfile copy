FROM nvidia/cuda:12.1.1-base-ubuntu20.04

WORKDIR /workspace

COPY requirements_cuda12.1.txt .

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    jq \
    libghc-yaml-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

ENV VIRTUAL_ENV=/opt/venv/dpimagebench
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN source activate

RUN apt-get update && apt-get install -y libmpich-dev

RUN pip install -r requirements_cuda12.1.txt

RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

COPY . /workspace
