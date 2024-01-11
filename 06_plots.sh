#!/bin/bash
# summarizing results and preparing plots

docker build ./analysis -t sdmtib/tracedsparql:analysis
docker run -it --rm -v ./results:/results sdmtib/tracedsparql:analysis
docker image rm sdmtib/tracedsparql:analysis
