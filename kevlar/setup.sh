#!/bin/bash

projectdir=$(dirname "BASH_SOURCE")
eigen3ver="3.4.0"
eigen3path="third_party/eigen-$eigen3ver"

mkdir -p third_party

# install Eigen
cd third_party &> /dev/null
if [ ! -d "$eigen3path" ]; then
    curl -O https://gitlab.com/libeigen/eigen/-/archive/$eigen3ver/eigen-$eigen3ver.tar.gz
    tar xzf eigen-$eigen3ver.tar.gz
    rm -f eigen-$eigen3ver.tar.gz
    cd eigen-$eigen3ver
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="." # installs into build directory
    make install
fi
cd ../../../ &> /dev/null
