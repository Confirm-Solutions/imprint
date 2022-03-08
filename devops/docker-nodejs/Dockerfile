FROM node:16.14.0-bullseye
LABEL maintainer="Gary Mulder"

EXPOSE 22
RUN apt update && \
    apt install -y openssh-server openjdk-11-jre && \
    /etc/init.d/ssh restart && \
    adduser --disabled-password --gecos "" jenkins && \
    mkdir /home/jenkins/.ssh
COPY id_rsa.pub /home/jenkins/.ssh/authorized_keys
RUN chown jenkins:jenkins /home/jenkins/.ssh/authorized_keys && \
    chmod 600 /home/jenkins/.ssh/authorized_keys
CMD ["/usr/sbin/sshd","-D"]
