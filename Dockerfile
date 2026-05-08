# Use Ubuntu 20.04 base image for better libssl1.1 support
FROM --platform=linux/amd64 ubuntu:20.04

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# Install necessary system dependencies and Rust
RUN apt-get update && \
    apt-get install -y \
    curl \
    clang \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 

# Install root dependencies
RUN apt-get update && \
    apt-get install -y \
    dpkg-dev \
    cmake \
    g++ \
    gcc \
    binutils \
    libx11-dev \
    libxpm-dev \
    libxft-dev \
    libxext-dev \
    python3 \
    libssl-dev \
    libafterimage0 \
    openssl

# Install rustfmt for the specified toolchain
#RUN rustup component add rustfmt --toolchain 1.76.0-aarch64-unknown-linux-gnu


# Install ROOT and CRES
COPY root_v6.30.04.Linux-ubuntu20.04-x86_64-gcc9.4.tar.gz /root/

# Install libssl1.1 from Ubuntu Focal repositories
RUN apt-get update && \
    apt-get install -y libssl1.1 && \
    rm -rf /var/lib/apt/lists/*

# Install cres with ntuple feature
RUN tar -xf /root/root_v6.30.04.Linux-ubuntu20.04-x86_64-gcc9.4.tar.gz -C /root \
    && cd /root/root && . bin/thisroot.sh && \
    cargo install --features ntuple --git https://github.com/a-maier/cres

# Set the entrypoint to run cres
# ENTRYPOINT ["cres"]
