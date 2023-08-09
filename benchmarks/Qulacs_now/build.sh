#!bin/bash

cd $(dirname $0)

apt-get update


CONTAINER_NAME=qulacs_now_container
IMAGE_NAME=qulacs_now_image

docker build -t ${IMAGE_NAME} .
docker run --name ${CONTAINER_NAME} --mount type=bind,source="$(pwd)",target=/home ${IMAGE_NAME} bash -c "sh build_by_docker.sh && nvcc -O2 -I /qulacs/include -L /qulacs/lib /home/main.cu -lvqcsim_static -lcppsim_static -lcsim_static -lgpusim_static -D _USE_GPU -lcublas -Xcompiler -fopenmp -o /home/main"
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}
docker image rm ${IMAGE_NAME}
