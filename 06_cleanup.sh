#!/bin/bash
# cleaning up the Docker environment

docker-compose down -v
docker image rm prohde/virtuoso-opensource-7:7.2.11 sdmtib/tracedsparql_engine:experiments
chown -R $(logname):$(logname) ./results ./error.log

