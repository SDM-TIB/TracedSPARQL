#!/bin/bash

# get current timestamp
timestamp() {
  date +%Y-%m-%dT%H-%M-%S
}

# error handling
error() {
  IFS='' read -r FIRSTLINE;
  if [ ! -z "$FIRSTLINE" ]
  then
    echo "${@}" >> error.log;
    echo "$FIRSTLINE" >> error.log;
  fi
  while IFS='' read -r line; do
    echo "$line" >> error.log;
  done;
};

iterations() {
  endpoint=$1
  query_dir=$2
  out_dir=$3
  shape_dir=$4
  api_config=$5

  for ((i=1;i<=RUNS;i++)); do
    for query_file in $(echo "./"$query_dir"/*.sparql"); do
      query_file=${query_file: 1}
      query_file_name=$(basename -- "$query_file")
      query_id=${query_file_name%.*}
      query_id=${query_id/-/.}

      docker exec -it $endpoint bash -c "isql-v 1111 dba dba 'EXEC=shutdown'" > /dev/null 2> /dev/null
      sleep 2s
      docker restart $endpoint $DOCKER_ENGINE > /dev/null
      until $(curl --output /dev/null --silent --head --fail http://localhost:${PORTS[$endpoint]}/sparql); do
          # the endpoint needs some time to be responsive, wait for a second if it is not yet up
          sleep 1s
      done

      if [ -z $api_config ];
        then
          echo $(timestamp) $endpoint "no_shapes" $query_id $i;
          error_hint=$(echo $(timestamp) $endpoint "no_shapes" $query_id $i);
	  output_dir=/results/raw/$out_dir/$endpoint/no_shapes/run_$i;
          mkdir -p "./"$output_dir
          docker exec -it $DOCKER_ENGINE bash -c "cd /TracedSPARQL; (timeout -s 15 602 python3 TracedSPARQL.py -q $query_file -c /TracedSPARQL/Config/${RDFMTS[$endpoint]} -i $query_id -p $output_dir);" > >(error $error_hint);
        else
          echo $(timestamp) $endpoint $shape_dir $query_id $api_config $i;
          error_hint=$(echo $(timestamp) $endpoint $shape_dir $query_id $api_config $i);
          output_dir=/results/raw/$out_dir/$endpoint/$(basename -- $shape_dir)/$api_config/run_$i;
          mkdir -p "./"$output_dir
          docker exec -it $DOCKER_ENGINE bash -c "cd /TracedSPARQL; (timeout -s 15 602 python3 TracedSPARQL.py -q $query_file -c /TracedSPARQL/Config/${RDFMTS[$endpoint]} -i $query_id -a /inputs/api_configs/$api_config.json -s /inputs/$shape_dir -p $output_dir);" > >(error $error_hint);
      fi
    done
  done
}

# run the specified test beds
run_testbeds() {
  declare -a endpoints=("${!1}")
  query_dir=$2
  declare -a shapes=("${!3}")
  declare -a configs=("${!4}")
  out_dir=$5

  # iterating over all endpoints
  for e in ${endpoints[@]}; do
    # if not ablation study: run all queries without validating SHACL
    [[ ! $out_dir =~ "ablation" ]] && iterations $e $query_dir $out_dir

    # run all queries for each combination of SHACL shape schema and API configuration
    for schema in ${shapes[@]}; do
      for config in ${configs[@]}; do
        iterations $e $query_dir $out_dir $schema $config
      done
    done

    # stop the endpoint
    docker stop $e > /dev/null
  done
}

