# Build this with 
# docker build -t vggt-test .

# Run docker container with
# docker run -it --gpus all vggt-test /bin/bash

# Then when ready to build and push to remote
# docker build -t gcr.io/tour-project-442218/vggt . && docker push gcr.io/tour-project-442218/vggt
# Then to run it
# docker run -it --gpus all gcr.io/tour-project-442218/vggt /bin/bash

# Once inside run
# gcloud storage rsync -r gs://tour_storage/data/tandt data/tandt
# TODO: Fill out the needed command
# gcloud storage rsync -r data/tandt/truck gs://tour_storage/data/tandt/truck

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install git
RUN apt-get update && apt-get install -y curl gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y \
    google-cloud-cli \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    ninja-build \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libflann-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install pixi using the official installation method
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"


# Copy the service account key file
COPY gcs-tour-project-service-account-key.json .

# Authenticate with Google Cloud
RUN gcloud auth activate-service-account --key-file=gcs-tour-project-service-account-key.json && \
    gcloud config set project tour-project-442218

# Update conda
RUN conda update -n base -c defaults conda -y
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc
RUN conda config --add channels defaults

# Make sure conda is in the PATH
ENV PATH /opt/conda/bin:$PATH