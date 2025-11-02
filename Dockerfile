FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    git curl pkg-config libpq-dev \
    socat \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN python -m pip install --no-cache-dir grpcio grpcio-tools && \
    mkdir -p /app/app/grpc_gen && \
    python -m grpc_tools.protoc \
      -I /app/api/protos \
      --python_out=/app/app/grpc_gen \
      --grpc_python_out=/app/app/grpc_gen \
      /app/api/protos/posts/posts.proto && \
    mkdir -p /app/app/grpc_gen/posts && \
    touch /app/app/grpc_gen/__init__.py /app/app/grpc_gen/posts/__init__.py

EXPOSE 50051
