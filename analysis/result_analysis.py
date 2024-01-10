#!/usr/bin/env python3

import os
from statistics import mean
from statistics import stdev as std

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import genfromtxt

base_path = '/results'
raw_path = os.path.join(base_path, 'raw')
summarized_path = os.path.join(base_path, 'summarized')
plot_path = os.path.join(base_path, 'plots_violin')

kgs = {
    'lubm': ['SKG', 'MKG', 'LKG'],
    'watdiv': ['500k', '10M', '100M'],
    'dbpedia': ['DBpedia']
}
palette = {
    'Baseline': '#EDAE49',
    'Baseline S2S': '#D1495B',
    'TracedSPARQL': '#30638E',
    'TracedSPARQL S2S': '#2A5B21'
}
approaches = ['Baseline', 'Baseline S2S', 'TracedSPARQL', 'TracedSPARQL S2S']

queries_lubm = 10
queries_watdiv = 18
queries_dbpedia = 20


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


def get_stats(benchmark, kg, network):
    if benchmark == 'dbpedia':
        dataset = 'tracedsparql_' + benchmark
    else:
        if benchmark == 'lubm':
            kg = kg.lower()
        dataset = 'tracedsparql_' + benchmark + '_' + kg
    stats_net1 = genfromtxt(os.path.join(summarized_path, benchmark, dataset, network, 'stats.csv'), delimiter=',', names=True, dtype=None, encoding='utf8')
    stats_pd = pd.DataFrame(stats_net1)
    stats_pd.loc[stats_pd['approach'] == 'baseline', 'approach'] = 'Baseline'
    stats_pd.loc[stats_pd['approach'] == 's2s_baseline', 'approach'] = 'Baseline S2S'
    stats_pd.loc[stats_pd['approach'] == 'tracedsparql', 'approach'] = 'TracedSPARQL'
    stats_pd.loc[stats_pd['approach'] == 's2s_tracedsparql', 'approach'] = 'TracedSPARQL S2S'
    return stats_pd


def violin_plot(benchmark, network, num_queries, filename, title):
    stats_new = pd.DataFrame(columns=['Engine', 'kg', 'execution_time'])
    for kg in kgs[benchmark]:
        stats_pd = get_stats(benchmark, kg, network)
        for approach in approaches:
            if benchmark == 'watdiv' or benchmark == 'dbpedia':
                if 'S2S' in approach:
                    continue
            approach_pd = stats_pd[stats_pd['approach'] == approach].reset_index()
            times = approach_pd['total_execution_time'].to_list()
            num_results = len(times)
            while num_results < num_queries:
                times.append(600.0)
                num_results += 1
            stats_new = pd.concat([stats_new, pd.DataFrame({'Engine': approach, 'execution_time': times, 'kg': kg})])

    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(data=stats_new, x='kg', y='execution_time', hue='Engine', palette=palette, saturation=.75, cut=0, inner='point', inner_kws={'color': 'r', 's': 32})
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, prop={'size': 10}, ncol=4)
    plt.title(title, fontsize=12, fontweight='bold', y=1.05)
    plt.xlabel('KG Size', fontweight='bold', fontsize=10)
    plt.ylabel('Execution Time', fontweight='bold', fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.92)
    plt.savefig(os.path.join(plot_path, filename))


def plot_results():
    plots = [
        {'benchmark': 'lubm', 'network': 'network1', 'num_queries': queries_lubm,
         'filename': 'violin_combined_lubm_network1.png',
         'title': 'Execution Times for LUBM validated with LUBM-network1'},
        {'benchmark': 'lubm', 'network': 'network2', 'num_queries': queries_lubm,
         'filename': 'violin_combined_lubm_network2.png',
         'title': 'Execution Times for LUBM validated with LUBM-network2'},
        {'benchmark': 'watdiv', 'network': 'network1', 'num_queries': queries_watdiv,
         'filename': 'violin_combined_watdiv_network1.png',
         'title': 'Execution Times for WatDiv validated with WatDiv-network1'},
        {'benchmark': 'watdiv', 'network': 'network2', 'num_queries': queries_watdiv,
         'filename': 'violin_combined_watdiv_network2.png',
         'title': 'Execution Times for WatDiv validated with WatDiv-network2'},
        {'benchmark': 'dbpedia', 'network': 'dbpedia', 'num_queries': queries_dbpedia,
         'filename': 'violin_combined_dbpedia.png',
         'title': 'Execution Times for DBpedia validated with the DBpedia shapes'}
    ]
    for plot in plots:
        violin_plot(**plot)


if __name__ == '__main__':
    os.makedirs(summarized_path, exist_ok=True)
    summarize()
    os.makedirs(plot_path, exist_ok=True)
    plot_results()

