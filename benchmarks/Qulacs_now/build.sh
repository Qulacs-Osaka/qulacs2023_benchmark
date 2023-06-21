#!bin/bash

cd $(dirname $0)

apt-get update

docker build -t qulacs_now .
docker run -it qulacs_now
