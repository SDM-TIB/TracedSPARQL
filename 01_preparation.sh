#!/bin/bash
# prepare the data and containers for the experiment

cd datasources

wget --progress=bar https://service.tib.eu/ldmservice/dataset/01e4ef01-5ffe-4fe7-9be0-aba08510fc41/resource/53cd1b9e-f546-4363-ae3f-3d9e6f9ec026/download/lubm.zip
unzip lubm.zip && rm lubm.zip

wget --progress=bar https://service.tib.eu/ldmservice/dataset/01e4ef01-5ffe-4fe7-9be0-aba08510fc41/resource/5c0af72d-99ab-4149-a7f3-f38dca08ee8d/download/watdiv.zip
unzip watdiv.zip && rm watdiv.zip

wget --progress=bar https://service.tib.eu/ldmservice/dataset/01e4ef01-5ffe-4fe7-9be0-aba08510fc41/resource/e2d09ca8-acc0-4638-9f46-0671b9569ae0/download/dbpedia.zip
unzip dbpedia.zip && rm dbpedia.zip

cd ..

docker-compose up -d --build > /dev/null
sleep 30s  # give the containers some time for initialization
docker-compose stop > /dev/null

