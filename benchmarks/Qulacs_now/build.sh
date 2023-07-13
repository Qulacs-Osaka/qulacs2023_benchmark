#!bin/bash

cd $(dirname $0)

apt-get update


CONTAINER_NAME=qulacs_now_container
IMAGE_NAME=qulacs_now_image

docker build -t ${IMAGE_NAME} .
docker run -it --name ${CONTAINER_NAME} --mount type=bind,source="$(pwd)",target=/home ${IMAGE_NAME}
