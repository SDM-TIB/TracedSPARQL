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
palette_ablation = {
    'no heuristics': '#264653',
    'Heuristic 1': '#2A9D8F',
    'Heuristic 2': '#8AB17D',
    'Heuristic 3': '#E9C46A',
    'Heuristic 4': '#F4A261',
    'all heuristics combined': '#E76F51'
}
palette_comparison = {
    'no validation': '#EDAE49',
    'TracedSPARQL': '#30638E',
}
approaches = ['Baseline', 'Baseline S2S', 'TracedSPARQL', 'TracedSPARQL S2S']
approaches_ablation = ['no heuristics', 'Heuristic 1', 'Heuristic 2', 'Heuristic 3', 'Heuristic 4', 'all heuristics combined']
approaches_comparison = ['no validation', 'TracedSPARQL']

queries_lubm = 10
queries_watdiv = 18
queries_dbpedia = 20

queries_ablation_lubm = 5
queries_ablation_watdiv = 10
queries_ablation_dbpedia = 10


def stdev(data):
    if len(data) < 2:
        return 0.0
    else:
        return std(data)


def summarize_testbed(benchmark, dataset, shape_schema, config=None):
    if config is not None:
        testbed_path = os.path.join(raw_path, benchmark, dataset, shape_schema, config)
    else:
        testbed_path = os.path.join(raw_path, benchmark, dataset, shape_schema)

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
    testbed_summarized_path = os.path.join(summarized_path, benchmark, dataset, shape_schema)
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
            outfile.write('test,approach,total_execution_time,total_execution_time_std,query_time,query_time_std,validation_time,validation_time_std,query_val_join_time,query_val_join_time_std\n')

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

            for shape_schema in sorted(os.listdir(dataset_dir)):
                schema_dir = os.path.join(dataset_dir, shape_schema)
                if not os.path.isdir(schema_dir):
                    continue

                if shape_schema == "no_shapes":
                    summarize_testbed(benchmark, dataset, shape_schema)
                else:
                    for config in sorted(os.listdir(schema_dir)):
                        config_dir = os.path.join(schema_dir, config)
                        if not os.path.isdir(config_dir):
                            continue

                        summarize_testbed(benchmark, dataset, shape_schema, config)


def get_stats(benchmark, kg, shape_schema):
    if benchmark == 'dbpedia':
        dataset = 'tracedsparql_' + benchmark
    else:
        if benchmark == 'lubm':
            kg = kg.lower()
        dataset = 'tracedsparql_' + benchmark + '_' + kg
    stats_net1 = genfromtxt(os.path.join(summarized_path, benchmark, dataset, shape_schema, 'stats.csv'), delimiter=',', names=True, dtype=None, encoding='utf8')
    stats_pd = pd.DataFrame(stats_net1)
    stats_pd.loc[stats_pd['approach'] == 'baseline', 'approach'] = 'Baseline'
    stats_pd.loc[stats_pd['approach'] == 's2s_baseline', 'approach'] = 'Baseline S2S'
    stats_pd.loc[stats_pd['approach'] == 'tracedsparql', 'approach'] = 'TracedSPARQL'
    stats_pd.loc[stats_pd['approach'] == 's2s_tracedsparql', 'approach'] = 'TracedSPARQL S2S'
    return stats_pd


def violin_plot(benchmark, shape_schema, num_queries, filename, title):
    stats_new = None
    for kg in kgs[benchmark]:
        stats_pd = get_stats(benchmark, kg, shape_schema)
        for approach in approaches:
            if benchmark == 'watdiv' or benchmark == 'dbpedia':
                if 'S2S' in approach:
                    continue
            approach_pd = stats_pd[stats_pd['approach'] == approach].reset_index()
            times = approach_pd['total_execution_time'].to_list()
            times.extend([600.0] * (num_queries - len(times)))
            current_stats = pd.DataFrame({'Engine': approach, 'execution_time': times, 'kg': kg})
            if stats_new is None:
                stats_new = current_stats
            else:
                stats_new = pd.concat([stats_new, current_stats])

    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(data=stats_new, x='kg', y='execution_time', hue='Engine', palette=palette, saturation=.75, cut=0, inner='point', inner_kws={'color': 'r', 's': 32})
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, prop={'size': 10}, ncol=4)
    plt.title(title, fontsize=12, fontweight='bold', y=1.05)
    plt.xlabel('Knowledge Graph Size', fontweight='bold', fontsize=10)
    plt.ylabel('Execution Time [s]', fontweight='bold', fontsize=10)
    ax.set_ylim(bottom=0, top=630, emit=True, auto=False)
    ax.axhline(y=600, color='r', linestyle=(0, (5, 10)), linewidth=1)
    plt.text(x=-0.475, y=605, s='Timeout', color='r')
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.92)
    plt.savefig(os.path.join(plot_path, filename))


def violin_ablation(dataset, shape_schema, num_queries, filename, title):
    stats = pd.DataFrame(genfromtxt(os.path.join(summarized_path, 'ablation', 'tracedsparql_' + dataset, shape_schema, 'stats.csv'), delimiter=',', names=True, dtype=None, encoding='utf8'))
    stats.loc[stats['approach'] == 'baseline', 'approach'] = 'no heuristics'
    stats.loc[stats['approach'] == 'tracedsparql', 'approach'] = 'all heuristics combined'
    stats.loc[stats['approach'] == 'opt-1', 'approach'] = 'Heuristic 1'
    stats.loc[stats['approach'] == 'opt-2', 'approach'] = 'Heuristic 2'
    stats.loc[stats['approach'] == 'opt-3', 'approach'] = 'Heuristic 3'
    stats.loc[stats['approach'] == 'opt-4', 'approach'] = 'Heuristic 4'

    stats_new = None
    for approach in approaches_ablation:
        df_approach = stats[stats['approach'] == approach].reset_index()
        times = df_approach['total_execution_time'].to_list()
        times.extend([600.0] * (num_queries - len(times)))
        current_stats = pd.DataFrame({'Heuristic': approach, 'execution_time': times})
        if stats_new is None:
            stats_new = current_stats
        else:
            stats_new = pd.concat([stats_new, current_stats])

    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(data=stats_new, x='Heuristic', y='execution_time', hue='Heuristic', palette=palette_ablation, saturation=.9, cut=0, inner='point', inner_kws={'color': 'r', 's': 32})
    plt.title(title, fontsize=12, fontweight='bold', y=1.05)
    plt.xlabel('Heuristic', fontweight='bold', fontsize=10)
    plt.ylabel('Execution Time [s]', fontweight='bold', fontsize=10)
    ax.set_ylim(bottom=0, top=630, emit=True, auto=False)
    ax.axhline(y=600, color='r', linestyle=(0, (5, 10)), linewidth=1)
    plt.text(x=-0.475, y=605, s='Timeout', color='r')
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.92)
    plt.savefig(os.path.join(plot_path, filename))


def get_stats_comparison(benchmark, kg, shape_schema, num_queries):
    if benchmark.lower() == 'dbpedia':
        dataset = 'tracedsparql_' + benchmark.lower()
    else:
        dataset = 'tracedsparql_' + benchmark.lower() + '_' + (kg.lower() if benchmark.lower() == 'lubm' else kg)
    stats = pd.DataFrame(genfromtxt(os.path.join(summarized_path, benchmark.lower(), dataset, shape_schema, 'stats.csv'), delimiter=',', names=True, dtype=None, encoding='utf8'))
    stats.loc[stats['approach'] == 'tracedsparql', 'approach'] = 'TracedSPARQL'
    stats = pd.concat([stats, pd.DataFrame(genfromtxt(os.path.join(summarized_path, benchmark.lower(), dataset, 'no_shapes', 'stats.csv'), delimiter=',', names=True, dtype=None, encoding='utf8'))])
    stats.loc[stats['approach'] == 'no_shacl', 'approach'] = 'no validation'

    stats_new = None
    for approach in approaches_comparison:
        df_approach = stats[stats['approach'] == approach].reset_index()
        times = df_approach['total_execution_time'].to_list()
        times.extend([600.0] * (num_queries - len(times)))
        current_stats = pd.DataFrame({
            'Validation': approach,
            'execution_time': times,
            'dataset': (benchmark + ' ' + kg) if benchmark.lower() != 'dbpedia' else benchmark
        })
        if stats_new is None:
            stats_new = current_stats
        else:
            stats_new = pd.concat([stats_new, current_stats])

    return stats_new


def violin_comparison():
    filename = 'violin_comparison.png'
    title = 'Comparison of execution times with and without validation'

    stats_lubm_skg = get_stats_comparison('LUBM', 'SKG', 'schema1', queries_lubm)
    stats_lubm_mkg = get_stats_comparison('LUBM', 'MKG', 'schema1', queries_lubm)
    stats_lubm_lkg = get_stats_comparison('LUBM', 'LKG', 'schema1', queries_lubm)
    stats_watdiv_500k = get_stats_comparison('WatDiv', '500k', 'schema1', queries_watdiv)
    stats_watdiv_10M = get_stats_comparison('WatDiv', '10M', 'schema1', queries_watdiv)
    stats_watdiv_100M = get_stats_comparison('WatDiv', '100M', 'schema1', queries_watdiv)
    stats_dbpedia = get_stats_comparison('DBpedia', 'dbpedia', 'dbpedia', queries_dbpedia)

    stats_new = pd.concat([stats_lubm_skg, stats_lubm_mkg, stats_lubm_lkg, stats_watdiv_500k, stats_watdiv_10M, stats_watdiv_100M, stats_dbpedia])
    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(data=stats_new, x='dataset', y='execution_time', hue='Validation', palette=palette_comparison, saturation=.75, cut=0, inner='point', inner_kws={'color': 'r', 's': 32}, split=True, log_scale=True, gap=0.1)
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, prop={'size': 10}, ncol=2)
    plt.title(title, fontsize=12, fontweight='bold', y=1.05)
    plt.xlabel('Knowledge Graph', fontweight='bold', fontsize=10)
    plt.ylabel('Execution Time [s] (log scale)', fontweight='bold', fontsize=10)
    ax.set_ylim(bottom=0.3, top=900, emit=True, auto=False)
    ax.axhline(y=600, color='r', linestyle=(0, (5, 10)), linewidth=1)
    plt.text(x=-0.45, y=650, s='Timeout', color='r')
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.92)
    plt.savefig(os.path.join(plot_path, filename))


def plot_results():
    plots = [
        {'benchmark': 'lubm', 'shape_schema': 'schema1', 'num_queries': queries_lubm,
         'filename': 'violin_combined_lubm_schema1.png',
         'title': 'Execution Times for LUBM validated with LUBM-schema1'},
        {'benchmark': 'lubm', 'shape_schema': 'schema2', 'num_queries': queries_lubm,
         'filename': 'violin_combined_lubm_schema2.png',
         'title': 'Execution Times for LUBM validated with LUBM-schema2'},
        {'benchmark': 'watdiv', 'shape_schema': 'schema1', 'num_queries': queries_watdiv,
         'filename': 'violin_combined_watdiv_schema1.png',
         'title': 'Execution Times for WatDiv validated with WatDiv-schema1'},
        {'benchmark': 'watdiv', 'shape_schema': 'schema2', 'num_queries': queries_watdiv,
         'filename': 'violin_combined_watdiv_schema2.png',
         'title': 'Execution Times for WatDiv validated with WatDiv-schema2'},
        {'benchmark': 'dbpedia', 'shape_schema': 'dbpedia', 'num_queries': queries_dbpedia,
         'filename': 'violin_combined_dbpedia.png',
         'title': 'Execution Times for DBpedia validated with the DBpedia shapes'}
    ]
    for plot in plots:
        violin_plot(**plot)

    ablation_plots = [
        {'dataset': 'lubm_mkg', 'shape_schema': 'schema2', 'num_queries': queries_ablation_lubm,
         'filename': 'violin_ablation_lubm.png',
         'title': 'Ablation Study for LUBM MKG validated with LUBM-schema2'},
        {'dataset': 'watdiv_10M', 'shape_schema': 'schema1', 'num_queries': queries_ablation_watdiv,
         'filename': 'violin_ablation_watdiv.png',
         'title': 'Ablation Study for WatDiv MKG validated with WatDiv-schema1'},
        {'dataset': 'dbpedia', 'shape_schema': 'dbpedia', 'num_queries': queries_ablation_dbpedia,
         'filename': 'violin_ablation_dbpedia.png',
         'title': 'Ablation Study for DBpedia validated with the DBpedia shapes'}
    ]
    for ablation_plot in ablation_plots:
        violin_ablation(**ablation_plot)

    violin_comparison()


if __name__ == '__main__':
    os.makedirs(summarized_path, exist_ok=True)
    summarize()
    os.makedirs(plot_path, exist_ok=True)
    plot_results()

