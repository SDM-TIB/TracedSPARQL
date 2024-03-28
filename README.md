[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Python version](https://img.shields.io/badge/python-3-blue.svg)
[![Data](https://img.shields.io/badge/Data-10.57702/wfl730bc-green.svg)](https://doi.org/10.57702/wfl730bc)

# TracedSPARQL

TracedSPARQL is tracing SHACL validations during SPARQL query processing towards a better understanding of SPARQL query results.

## Table of Contents
1. [Preparation of the Environment](#preparation-of-the-environment)
    1. [Machine Requirements](#machine-requirements)
    1. [Software](#software)
    1. [Bash Commands](#bash-commands)
1. [Experiments](#experiments)
    1. [Research Questions](#research-questions)
    1. [Data](#data)
    1. [Engines](#engines)
    1. [Setups](#setups)
    1. [How to reproduce?](#how-to-reproduce)
    1. [Results](#results)
1. [License](#license)
1. [References](#references)

## Preparation of the Environment
### Machine Requirements
- OS: Ubuntu 16.04.6 LTS or newer
- Memory: 128 GiB
- HDD: approx. 50 GiB free disk space

### Software
- Docker - v19.03.6 or newer
- docker-compose - v1.26.0 or newer

### Bash Commands
The experiment scripts use the following bash commands:

- basename
- cd
- chown
- declare (with options -a and -A)
- echo
- logname
- rm
- sleep
- source
- unzip
- wget

## Experiments
### Research Questions
1. What is the overhead of adding online SHACL validation to the SPARQL query processing?
1. Do the proposed optimizations increase the performance?
1. Which heuristic has the highest single effect?

### Data & SHACL Shape Schemas
Data from three benchmarks are used in the evaluation of TracedSPARQL.
The following benchmarks are covered:

- Lehigh University Benchmark (LUBM) [\[1\]](#1)
- Waterloo SPARQL Diversity Test Suite (WatDiv) [\[2\]](#2)
- DBpedia [\[3\]](#3)

For LUBM and WatDiv knowledge graphs of three different sizes are used.
Hence, a total of seven knowledge graphs are evaluated.
For LUBM and WatDiv two SHACL shape schemas of different complexity are validated.
In the case of DBpedia, a single SHACL shape schema is used.
10 SPARQL queries from the LUBM benchmark are included in the evaluation.
From WatDiv, 18 SPARQL queries are considered.
20 SPARQL queries are created for the evaluation of DBpedia.
The SPARQL queries cover at least one SHACL shape schema of the respective benchmark.
All data used are made public [\[4\]](#4).

### Engines
TracedSPARQL is compared with a naive approach, referred to as _baseline_.
The federated SPARQL query engine used is DeTrusty [\[5\]](#5).
The SHACL validation is performed by Trav-SHACL [\[6\]](#6) and SHACL2SPARQLpy [\[7\]](#7), a Python implementation of SHACL2SPARQL [\[8\]](#8).
This leads to the following engines included in the evaluation:

| Name             | SHACL Validator | Heuristics |
|------------------|-----------------|------------|
| Baseline         | Trav-SHACL      | none       |
| Baseline S2S     | SHACL2SPARQLpy  | none       |
| TracedSPARQL     | Trav-SHACL      | all        |
| TracedSPARQL S2S | SHACL2SPARQLpy  | all        |

### Setups
The combination of a knowledge graph, engine, SHACL shape schema, and SPARQL query is referred to as a testbed; this leads to a total of 1,065 testbeds.
Each testbed is executed five times.
Caches are flushed between the execution of two consecutive testbeds.

### How to reproduce?
In order to facilitate the reproduction of the results, all components are encapsulated in Docker containers and the experiments are controlled via Shell scripts.
You can run the entire pipeline by executing:
```bash
sudo ./00_auto.sh
```

In the following, the different scripts are described in short.

- _00_auto.sh_: Executes the entire experiment automatically
- _01_preparation.sh_: Prepares the experimental environment, i.e., downloads the data and sets up the Docker containers
- _02_experiments_lubm.sh_: Executes the experiments for LUBM
- _03_experiments_watdiv.sh_: Executes the experiments for WatDiv
- _04_experiments_dbpedia.sh_: Executes the experiments for DBpedia
- _05_ablation_study.sh_: Executes the ablation study
- _06_plots.sh_: Creates the plots presented in the paper
- _07_cleanup.sh_: Cleans up the experimental environment including changing the ownership of result files to the user executing the script
- _run_testbeds.sh_: Contains functions for performing the experiments
- _variables.sh_: Contains variables used for performing the experiments

### Results

The result plots included in the paper and a brief summary is available in the [results directory](results/README.md).

## License
TracedSPARQL is licensed under GPL-3.0, see the [license](https://github.com/SDM-TIB/TracedSPARQL/blob/master/LICENSE).

## References
<a name="1">[1]</a> Y. Guo, Z. Pan, J. Heflin. _LUBM: A Benchmark for OWL Knowledge Base Systems_. Journal of Web Semantics 3(2-3), 158-182 (2005). DOI: [10.1016/j.websem.2005.06.005](https://doi.org/10.1016/j.websem.2005.06.005)

<a name="2">[2]</a> G. Aluç, O. Hartig, M.T. Özsu, K. Daudjee. _Diversified Stress Testing of RDF Data Management Systems_. In: The Semantic Web -- ISWC 2014, Springer, Cham, 2014. DOI: [10.1007/978-3-319-11964-9_13](https://doi.org/10.1007/978-3-319-11964-9_13)

<a name="3">[3]</a> S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, Z. Ives. _DBpedia: A Nucleus for a Web of Open Data_. In: The Semantic Web, Springer, Berlin, Heidelberg, 2007. DOI: [10.1007/978-3-540-76298-0_52](https://doi.org/10.1007/978-3-540-76298-0_52)

<a name="4">[4]</a> P.D. Rohde, M.-E. Vidal. _Dataset: TracedSPARQL Benchmarks_. Leibniz Data Manager (2023). DOI: [10.57702/wfl730bc](https://doi.org/10.57702/wfl730bc)

<a name="5">[5]</a> P.D. Rohde, M. Bechara, Avellino. _DeTrusty v0.15.0_. Zenodo (2023). DOI: [https://doi.org/10.5281/zenodo.10245898](https://doi.org/10.5281/zenodo.10245898).

<a name="6">[6]</a> M. Figuera, P.D. Rohde, M.-E. Vidal. _Trav-SHACL: Efficiently Validating Networks of SHACL Constraints_. In: The Web Conference, ACM, New York, NY, USA, 2021. DOI: [10.1145/3442381.3449877](https://doi.org/10.1145/3442381.3449877).

<a name="7">[7]</a> M. Figuera, P.D. Rohde. _SHACL2SPARQLpy v1.3.0_. GitHub (2023). URL: [https://github.com/SDM-TIB/SHACL2SPARQLpy](https://github.com/SDM-TIB/SHACL2SPARQLpy)

<a name="8">[8]</a> J. Corman, F. Florenzano, J.L. Reutter, O. Savković. _SHACL2SPARQL: Validating a SPARQL Endpoint against Recursive SHACL Constraints_. In: Proceedings of the ISWC 2019 Satellite Tracks, CEUR-WS, Aachen, Germany, 2019. URL: [https://ceur-ws.org/Vol-2456/paper43.pdf](https://ceur-ws.org/Vol-2456/paper43.pdf)

