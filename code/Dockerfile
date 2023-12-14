FROM python:3.12.1-slim-bookworm

# install dependencies
COPY requirements.txt /TracedSPARQL/requirements.txt
RUN python -m pip install --upgrade --no-cache-dir pip==23.3.* && \
    python -m pip install --no-cache-dir -r /TracedSPARQL/requirements.txt

# copy the source code into the container
COPY . /TracedSPARQL
RUN cd /TracedSPARQL && mkdir -p Config
WORKDIR /TracedSPARQL

# keep the container alive
CMD ["tail", "-f", "/dev/null"]
