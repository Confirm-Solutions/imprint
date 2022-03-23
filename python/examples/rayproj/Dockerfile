FROM rayproject/ray AS kevlar
LABEL maintainer="Gary Mulder"

EXPOSE 8265
USER $RAY_UID
ENV HOME=/home/ray 
ENV GIT_SSH_COMMAND="ssh -vvv"
COPY id_rsa.docker id_rsa.docker

# Build env for Kevlar
RUN sudo \
    id && \
    apt update && \
    apt install -y git g++-9 cmake curl && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10

# Install Python 3.x and latest Ray
RUN id && \
    conda install -y python=3.9 && \
    conda install -y grpcio==1.42 && \
    pip install ray==1.11.0
 
# Build and install Kevlar
ENV GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -i id_rsa.docker -o IdentitiesOnly=yes'
RUN id && \
    chmod 700 id_rsa.docker && \
    git clone --branch gary.mulder/revamp_smoke_tests git@github.com:mikesklar/kevlar.git && \
    cd kevlar/src/kevlar/ && \
    bash -x ./setup.sh && \
    cd ../../python/ && \
    pip install --user -vvv -e .

#CMD /home/ray/anaconda3/bin/ray start --address='172.17.0.2:6379' --redis-password='5241590000000000' --num-cpus=8 && \
#    tail -F /tmp/ray/session_latest/logs/log_monitor.log
