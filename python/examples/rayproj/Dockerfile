FROM rayproject/ray
LABEL maintainer="Gary Mulder"

ENV BRANCH="gary.mulder/ray_cluster"
ENV REPO="git@github.com:mikesklar/kevlar.git"
USER ray
ENV HOME="/home/ray"
WORKDIR $HOME

# Build env for Kevlar
ENV EXTRA_PKGS="gcc-9 g++-9 curl"
RUN sudo apt update && \
    sudo apt install -y $EXTRA_PKGS && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10 

# Install Python 3.x and Ray
# Note: latest grpcio breaks Ray
RUN conda install -y python=3.9 && \
    conda install -y grpcio==1.42 && \
    conda clean -y --all && \
    pip install --no-cache-dir ray==1.11.0
 
# Get latest Kevlar src
RUN mkdir "$HOME/.ssh" 
COPY id_rsa "$HOME/.ssh/id_rsa"
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
ENV CC="gcc"
ENV CXX="g++"
RUN sudo chown -R ray "$HOME/.ssh" && \
    chmod 700 "$HOME/.ssh/id_rsa" && \
    git clone --branch $BRANCH $REPO && \
    cd kevlar/ && \
    ./generate_bazelrc

# Bazel build and install for pykevlar
ENV PATH="$PATH:$HOME/bin"
RUN mkdir "$HOME/bin" && \
    curl -Lo "$HOME/bin/bazel" https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64 && \
    chmod +x "$HOME/bin/bazel" && \
    cd kevlar/python/ && \
    #pip install --no-cache-dir -r requirements.txt && \
    bazel build --config=gcc //python:pykevlar_wheel && \
    pip install --no-cache-dir -v --user `bazel info bazel-genfiles`/python/dist/pykevlar*.whl

# Clean up cached data and kevlar build
RUN sudo sudo apt-get autoremove -y $EXTRA_PKGS && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo apt-get clean

ENTRYPOINT ["ray", "start", "--block"]
