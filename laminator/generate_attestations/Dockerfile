# Use the Gramine project Docker image
FROM gramineproject/gramine

ENV DEBIAN_FRONTEND=noninteractive

COPY . /generate_attestations

WORKDIR /generate_attestations

ENV PATH="myenv/bin:$PATH"
RUN . myenv/bin/activate


RUN apt-get update && apt-get install -y build-essential \
    autoconf bison gawk nasm ninja-build pkg-config python3 python3-click \
    python3-jinja2 python3-pip python3-pyelftools wget \
    bc

# Install Python 3.9 and pip
RUN apt-get install -y python3 python3-pip

RUN which pip


# Upgrade pip to the latest version for Python 3.9
RUN python3 -m pip install --upgrade --break-system-packages pip

RUN python3 -m pip install 'meson>=0.56' 'tomli>=1.1.0' 'tomli-w>=0.4.0'

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN apt-get install -y wget gdebi-core

# Copy the CIFAR-10 folder into the container
#RUN wget https://packages.microsoft.com/ubuntu/20.04/prod/pool/main/a/az-dcap-client/az-dcap-client_1.12.0_amd64.deb

# Install the .deb package
#RUN gdebi --non-interactive az-dcap-client_1.12.0_amd64.deb

RUN /restart_aesm.sh

# Copy the  folder into the container
RUN gramine-sgx-gen-private-key

RUN pip install -r ./requirements.txt
