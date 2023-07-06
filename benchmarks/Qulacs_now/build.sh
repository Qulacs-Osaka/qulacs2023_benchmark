#!bin/bash

cd $(dirname $0)

apt-get update

path=$(pwd)

docker build -t qulacs_now .
docker run --mount type=bind,source=$path,target=/home -it qulacs_now
