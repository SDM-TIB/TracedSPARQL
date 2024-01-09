FROM python:3.9.18-slim-bookworm

# copy the files and install dependencies
COPY . /analysis
RUN python -m pip install --upgrade --no-cache-dir pip==23.3.* && \
    python -m pip install --no-cache-dir -r /analysis/requirements.txt && \
    chmod u+x /analysis/result_analysis.py

WORKDIR /analysis

# summarize the results
ENTRYPOINT ["./result_analysis.py"]
