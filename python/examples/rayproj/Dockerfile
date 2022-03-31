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
    sudo apt install -y $EXTRA_PKGS

# Install Python 3.x and Ray
# Note: latest grpcio breaks Ray
RUN id && \
    conda install -y python=3.9 && \
    conda install -y grpcio==1.42 && \
    conda clean -y --all && \
    pip install --no-cache-dir ray==1.11.0
 
# Get latest Kevlar src
RUN mkdir "$HOME/.ssh" 
COPY id_rsa "$HOME/.ssh/id_rsa"
#ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -i $HOME/id_rsa.github -o IdentitiesOnly=yes"
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
RUN id && \
    sudo chown -R ray "$HOME/.ssh" && \
    chmod 700 "$HOME/.ssh/id_rsa" && \
    git clone --branch $BRANCH $REPO 

# Bazel build for Kevlar
ENV CC="/usr/bin/gcc-9"
ENV CXX="/usr/bin/g++-9"
ENV PATH="$PATH:$HOME/bin"
RUN mkdir "$HOME/bin" && \
    curl -Lo "$HOME/bin/bazel" https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64 && \
    chmod +x "$HOME/bin/bazel" && \
    cd kevlar/python/ && \
    git status && \
    bazel fetch //kevlar:kevlar

# Build and install pykevar
RUN cd kevlar/python/ && \
    export CPLUS_INCLUDE_PATH="`bazel info output_base`/external/eigen" && \
    pip install --no-cache-dir -vv --user -e .

# Clean up cached data and kevlar build
RUN sudo sudo apt-get autoremove -y gcc-9 g++-9 curl && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo apt-get clean

ENTRYPOINT ["ray", "start", "--block"]
