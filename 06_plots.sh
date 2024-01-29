#!/bin/bash
# summarizing results and preparing plots

docker build ./analysis -t sdmtib/tracedsparql:analysis
rm -rf results/plots_violin
rm -rf results/summarized
docker run -it --rm -v ./results:/results sdmtib/tracedsparql:analysis
docker image rm sdmtib/tracedsparql:analysis
