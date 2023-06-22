#!bin/bash

cd $(dirname $0)

docker build -t qulacs_now .
docker run -rm -it -v ./:home/ qulacs_now
