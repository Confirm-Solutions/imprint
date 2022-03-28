FROM rayproject/ray AS kevlar
LABEL maintainer="Gary Mulder"

ENV BRANCH="gary.mulder/ray_cluster"
ENV REPO="git@github.com:mikesklar/kevlar.git"
EXPOSE 8265
USER ray
ENV HOME="/home/ray"
WORKDIR $HOME

# Build env for Kevlar
RUN sudo apt update && \
    sudo apt install -y git gcc-9 g++-9 curl

# Install Python 3.x and Ray
# Note: latest grpcio breaks Ray
RUN id && \
    conda install -y python=3.9 && \
    conda install -y grpcio==1.42 && \
    pip install ray==1.11.0
 
# Get latest Kevlar src
COPY id_rsa.docker "$HOME/id_rsa.docker"
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -i $HOME/id_rsa.docker -o IdentitiesOnly=yes"
RUN id && \
    sudo chown ray "$HOME/id_rsa.docker" && \
    chmod 700 "$HOME/id_rsa.docker" && \
    git clone --branch $BRANCH $REPO

# Bazel build for Kevlar
ENV CC="/usr/bin/gcc-9"
ENV CXX="/usr/bin/g++-9"
ENV PATH="$PATH:$HOME/bin"
RUN mkdir "$HOME/bin" && \
    curl -Lo "$HOME/bin/bazel" https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64 && \
    chmod +x "$HOME/bin/bazel" && \
    cd kevlar/python/ && \
    bazel fetch //kevlar:kevlar

# Build and install pykevar
RUN cd kevlar/python/ && \
    export CPLUS_INCLUDE_PATH="`bazel info output_base`/external/eigen" && \
    pip install --user -e .

#CMD /home/ray/anaconda3/bin/ray start --address='172.17.0.2:6379' --redis-password='5241590000000000' --num-cpus=8 && \
#    tail -F /tmp/ray/session_latest/logs/log_monitor.log
