#! /bin/bash
#run it from repo root dir
#docker build -t table_ocr:v1 -f data_extraction/docker . 

docker run -it \
        --rm \
        -p 8888:8888 \
        --name="data_extraction" \
        table_ocr:v1