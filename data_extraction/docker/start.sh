#! /bin/bash
#run it from repo root dir
#docker build -t data_extraction:v1 -f data_extraction/docker/Dockerfile . 

docker run -it \
        --rm \
        -p 8888:8888 \
        --name="data_extraction" \
        data_extraction