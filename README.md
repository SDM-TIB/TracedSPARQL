[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

# TracedSPARQL

TracedSPARQL is tracing SHACL validations during SPARQL query processing towards a better understanding of SPARQL query results.

## Preparation of the Environment
### Machine Requirements
- OS: Ubuntu 16.04.6 LTS or newer
- Memory: 128 GiB

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
### Data
Data from three benchmarks are used in the evaluation of TracedSPARQL.
The following benchmarks are covered:

- Lehigh University Benchmark (LUBM) [\[1\]](#1)
- Waterloo SPARQL Diversity Test Suite (WatDiv) [\[2\]](#2)
- DBpedia [\[3\]](#3)

For LUBM and WatDiv knowledge graphs of three different sizes are used.
Hence, a total of seven knowledge graphs are evaluated.
All data used are made public [\[4\]](#4).

## References
<a name="1">[1]</a> Y. Guo, Z. Pan, J. Heflin. _LUBM: A Benchmark for OWL Knowledge Base Systems_. Journal of Web Semantics 3(2-3), 158-182 (2005). DOI: [10.1016/j.websem.2005.06.005](https://doi.org/10.1016/j.websem.2005.06.005)

<a name="2">[2]</a> G. Aluç, O. Hartig, M.T. Özsu, K. Daudjee. _Diversified Stress Testing of RDF Data Management Systems_. In: The Semantic Web -- ISWC 2014, Springer, Cham, 2014. DOI: [10.1007/978-3-319-11964-9_13](https://doi.org/10.1007/978-3-319-11964-9_13)

<a name="3">[3]</a> S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, Z. Ives. _DBpedia: A Nucleus for a Web of Open Data_. In: The Semantic Web, Springer, Berlin, Heidelberg, 2007. DOI: [10.1007/978-3-540-76298-0_52](https://doi.org/10.1007/978-3-540-76298-0_52)

<a name="4">[4]</a> P.D. Rohde, M.-E. Vidal. _Dataset: TracedSPARQL Benchmarks_. Leibniz Data Manager (2023). DOI: [10.57702/wfl730bc](https://doi.org/10.57702/wfl730bc)

