# VenSyS-Clustering

VenSyS-Clustering is a Python package for creating representative ventilation load cases from room-level building data. It supports deterministic, norm-based load-case generation as well as distribution-based and sampling-based approaches for uncertain occupancy.

The package was developed for ventilation system planning workflows in which detailed hourly demand information has to be reduced to a smaller set of representative scenarios. The resulting load cases can be exported as YAML files and used in downstream optimisation models.

## Features

- Compute required room and zone volume flows from building data, occupancy profiles, and normative airflow requirements.
- Create deterministic load cases by clustering time slots based on required volume flows.
- Add optional revision volume flows and theoretical maximum load cases.
- Build discrete probability mass functions for room or zone volume-flow demand based on binomial occupancy assumptions.
- Cluster time slots using Wasserstein distances between demand distributions.
- Generate sampled load cases from room and zone distributions using quasi-Monte-Carlo sampling with Sobol sequences.
- Export and import load-case data in YAML format.

## Installation

From the project root:

```bash
pip install .
```

To install the package directly from GitHub:

```bash
pip install git+https://github.com/jhpb7/vensys-clustering
```

To install the listed dependencies explicitly:

```bash
pip install -r requirements.txt
```

Python 3.8 or newer is required.

## Repository structure

```text
src/vensys_clustering/
  __init__.py
  utils.py                  # deterministic, norm-based load-case generation
  load_distributions.py     # probability distributions and Wasserstein clustering
  load_case_sampling.py     # Monte Carlo and Sobol-based load-case sampling
  data/general.yml          # general normative input data

examples/
  create_profile_based_load_cases.ipynb
  create_distribution_based_load_cases.ipynb
  create_monte_carlo_load_cases.ipynb
  compare_approaches.ipynb
  input_files/
  output_files/

requirements.txt
setup.py
```

The notebooks in `examples/` show complete workflows for the available load-case generation methods.

## Input data

The package combines general normative data with building-specific input data.

The file `src/vensys_clustering/data/general.yml` contains general data such as room categories, occupancy profiles, per-person airflow rates, and area-based airflow rates.

Building-specific input data are provided as YAML files or Python dictionaries. They typically contain the rooms of the building, the room type, floor area, maximum number of persons, and optional mappings for merging rooms into larger zones.

## Load-case generation methods

### Deterministic load cases

The deterministic workflow computes required volume flows for each room and time slot and clusters the resulting demand vectors. The representative load cases are described by mean volume flows and optional quantiles. This approach is suitable when the normative occupancy profiles are used directly as deterministic input data.

### Distribution-based load cases

The distribution-based workflow represents room demand as discrete probability mass functions. These distributions are based on binomial occupancy assumptions and include the resulting volume-flow demand from person-based and area-based requirements.

Time slots can be compared using the Wasserstein distance between room or zone demand distributions. This allows clustering based on the full demand distribution instead of only clustering point estimates such as mean values.

### Sampled load cases

The sampling workflow draws load cases from the room or zone demand distributions. The current implementation supports Sobol-based quasi-Monte-Carlo sampling. Identical sampled states are counted and converted into load cases with corresponding time shares.

This approach can be used to generate Monte Carlo-style load-case sets from the probabilistic occupancy model.

## Outputs

The generated output is a YAML-compatible dictionary containing load cases and time shares. Depending on the workflow, each load case contains room- or zone-level volume-flow values such as means, quantiles, or sampled demand values.

Example output files are provided in `examples/output_files/`.

## Typical workflow

A typical workflow consists of the following steps:

1. Load the general data and building-specific input data.
2. Compute required room-level volume flows.
3. Optionally merge rooms into zones.
4. Generate load cases using one of the available methods.
5. Export the resulting load cases to YAML.

The example notebooks provide runnable versions of these workflows.

## Notes

Keep the volume-flow units consistent across all input files. The package assumes that the normative input data and building data use compatible units.

For Sobol sampling, powers of two are recommended for the number of samples because this improves the balance properties of the Sobol sequence.

## Note on AI usage

Parts of this repository, including documentation and code snippets, were prepared with the assistance of AI-based tools. All outputs were reviewed, validated, and adapted by the author.

## Funding

The code was developed within Julius Breuer's dissertation *Algorithmische Systemplanung raumlufttechnischer Anlagen* at TU Darmstadt. Parts of the dissertation results were obtained within the research project *Algorithmic System Planning of Air Handling Units*, Project No. 22289 N/1, funded by the program for promoting Industrial Collective Research (IGF) of the German Federal Ministry for Economic Affairs and Climate Action (BMWK), approved by the Deutsches Zentrum für Luft- und Raumfahrt (DLR). The author thanks the participants of the working group for their constructive collaboration.

## Citation

Citation information will be added later.
