#!/usr/bin/env python3

import os
from statistics import mean
from statistics import stdev as std

from numpy import genfromtxt

base_path = '/results'
raw_path = os.path.join(base_path, 'raw')
summarized_path = os.path.join(base_path, 'summarized')


def stdev(data):
    if len(data) < 2:
        return 0.0
    else:
        return std(data)


def summarize_testbed(benchmark, dataset, network, config=None):
    if config is not None:
        testbed_path = os.path.join(raw_path, benchmark, dataset, network, config)
    else:
        testbed_path = os.path.join(raw_path, benchmark, dataset, network)

    metrics = {}
    stats = {}

    for run in sorted(os.listdir(testbed_path)):
        run_dir = os.path.join(testbed_path, run)
        if not os.path.isdir(run_dir) or \
                not os.path.isfile(os.path.join(run_dir, 'metrics.csv')) or \
                not os.path.isfile(os.path.join(run_dir, 'stats.csv')):
            continue

        print(run_dir)
        data = genfromtxt(os.path.join(run_dir, 'metrics.csv'),
                          delimiter=',', names=True, dtype=None, encoding='utf8', ndmin=1)
        for entry in data:
            query = entry[0]
            tfft = entry[2]
            totaltime = entry[3]
            comp = entry[4]
            if query not in metrics.keys():
                metrics[query] = {
                    'tfft': [tfft],
                    'totaltime': [totaltime],
                    'comp': [comp]
                }
            else:
                metrics[query]['tfft'].append(tfft)
                metrics[query]['totaltime'].append(totaltime)
                metrics[query]['comp'].append(comp)

        data = genfromtxt(os.path.join(run_dir, 'stats.csv'),
                          delimiter=',', names=True, dtype=None, encoding='utf8', ndmin=1)
        for entry in data:
            query = entry[0]
            total_time = entry[2]
            query_time = entry[3]
            val_time = entry[4]
            join_time = entry[5]
            if query not in stats.keys():
                stats[query] = {
                    'total_time': [total_time],
                    'query_time': [query_time],
                    'val_time': [val_time],
                    'join_time': [join_time]
                }
            else:
                stats[query]['total_time'].append(total_time)
                stats[query]['query_time'].append(query_time)
                stats[query]['val_time'].append(val_time)
                stats[query]['join_time'].append(join_time)

    # data for all runs gathered, summarize now
    testbed_summarized_path = os.path.join(summarized_path, benchmark, dataset, network)
    os.makedirs(testbed_summarized_path, exist_ok=True)

    testbed_metrics_summarized_file = os.path.join(testbed_summarized_path, 'metrics.csv')
    new_file = False
    if not os.path.isfile(testbed_metrics_summarized_file):
        new_file = True

    with open(testbed_metrics_summarized_file, 'a', encoding='utf8') as outfile:
        if new_file:
            outfile.write('test,approach,tfft,totaltime,comp\n')

        for query in metrics.keys():
            entry = metrics[query]
            tfft_mean = str(mean(entry['tfft']))
            totaltime_mean = str(mean(entry['totaltime']))
            comp_mean = str(mean(entry['comp']))
            outfile.write(query + ',' + (config if config is not None else 'no_shacl') + ',' + tfft_mean + ',' + totaltime_mean + ',' + comp_mean + '\n')

    testbed_stats_summarized_file = os.path.join(testbed_summarized_path, 'stats.csv')
    new_file = False
    if not os.path.isfile(testbed_stats_summarized_file):
        new_file = True

    with open(testbed_stats_summarized_file, 'a', encoding='utf8') as outfile:
        if new_file:
            outfile.write('test,approach,total_execution_time,total_execution_time_std,query_time,query_time_std,network_validation_time,network_validation_time_std,query_val_join_time,query_val_join_time_std\n')

        for query in stats.keys():
            entry = stats[query]
            total_time_mean = str(mean(entry['total_time']))
            total_time_std = str(stdev(entry['total_time']))
            query_time_mean = str(mean(entry['query_time']))
            query_time_std = str(stdev(entry['query_time']))
            val_time_mean = str(mean(entry['val_time']))
            val_time_std = str(stdev(entry['val_time']))
            join_time_mean = str(mean(entry['join_time']))
            join_time_std = str(stdev(entry['join_time']))
            outfile.write(query + ',' + (config if config is not None else 'no_shacl') + ',' + total_time_mean + ',' + total_time_std + ',' + query_time_mean + ',' + query_time_std + ',' + val_time_mean + ',' + val_time_std + ',' + join_time_mean + ',' + join_time_std + '\n')

    del stats
    del metrics


def summarize():
    for benchmark in sorted(os.listdir(raw_path)):
        benchmark_dir = os.path.join(raw_path, benchmark)
        if not os.path.isdir(benchmark_dir):
            continue

        for dataset in sorted(os.listdir(benchmark_dir)):
            dataset_dir = os.path.join(benchmark_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            for network in sorted(os.listdir(dataset_dir)):
                network_dir = os.path.join(dataset_dir, network)
                if not os.path.isdir(network_dir):
                    continue

                if network == "no_shapes":
                    summarize_testbed(benchmark, dataset, network)
                else:
                    for config in sorted(os.listdir(network_dir)):
                        config_dir = os.path.join(network_dir, config)
                        if not os.path.isdir(config_dir):
                            continue

                        summarize_testbed(benchmark, dataset, network, config)


if __name__ == '__main__':
    os.makedirs(summarized_path, exist_ok=True)
    summarize()

