# `VenSyS-Clustering`: Ventilation System Scenario Reduction Toolkit

A Python toolkit for constructing **ventilation demand scenarios** from room-level data.  
It reduces the complexity of hourly load profiles by clustering operating hours into a small number of representative **load cases**. Each case is summarized by mean and quantile airflows, and a **theoretical maximum** load case can be added for robustness.
The occupancy profiles per building type are taken from DIN V 16798-1 and the airflow rates come from DIN EN 18599-10.




---

## Features

- 🏢 **Room & zone airflow calculation** from occupancy profiles, per-person, and per-area requirements.  
- ⏱ **Time-slot clustering** into representative load cases.  
- 📊 **Scenario summaries** with means and optional quantiles (e.g., q95).  
- 📄 **Export/import** scenarios via YAML.  
- 🎓 Two complementary methods:
  - **Deterministic (norm-based):** K-Means clustering on required flows.  
    - Add a **maximal load case** with zero frequency.  
    - Add a **revision volume flows**.
  - **Probabilistic (distribution-based):** Wasserstein distances between per-zone PMFs. 
    - Add **zero-inflation** to account for days when individual rooms are empty.


---

## Installation

# from the project root
```bash
pip install .
```
If you want to directly install from GitHub (without cloning):
```bash
pip install git+https://github.com/jhpb7/vensys-clustering
```
To install all subpackages use
```bash
pip install -r requirements.txt
```

Python ≥ 3.8 required

## Repository Layout
```
src/vensys-clustering/
  __init__.py
  utils.py                # deterministic (norm-based) pipeline
  wasserstein.py          # probabilistic (distribution-based) pipeline
  data/general.yml        # example normative data

examples/
  create_profile_based_load_cases.ipynb
  create_distribution_based_load_cases.ipynb
  compare_approaches.ipynb
  input_files/
  output_files/

requirements.txt
setup.py
```

See examples/ for end-to-end workflows.

Example input files are in examples/input_files/.

Example YAML outputs are in examples/output_files/.


## Input Data
Data from the standards (src/vensys-clustering/data/general.yml)

Contains:

    Room categories (e.g., classroom, office, sanitary).

    Hourly occupancy profiles.

    Per-person airflow (m³/h/person).

    Per-area airflow (m³/h/m²).

Building data (user-provided)

A Python dict or YAML with:

    List of rooms.

    Mapping roomtype, area, max_num_person.

    (Optional) rooms_to_merge for zone aggregation.

    (Optional) q_revision_tot for enforced minima.

## Outputs

A YAML file with:

    Load cases (per cluster and zone: mean, q95, …).

    Time shares (fraction of operating hours).

    Optional max case with frequency 0%.

See examples/output_files/ for concrete results.

### Tips
    Use analyze_cluster_quality or silhouette scores to choose the number of clusters.

    Keep flow units consistent (m³/h).

    The max load case ensures robustness.

## Note on AI usage
Parts of this repository (documentation and/or code snippets) were prepared with the assistance of AI-based tools, namely ChatGPT version 4 and 5. All outputs were reviewed, validated, and adapted by the authors.


## Funding
The presented code was written within Julius Breuer's dissertation ‘‘Algorithmische Systemplanung raumlufttechnischer Anlagen’’ at TU Darmstadt. Some of the results of the dissertation were obtained within the research project ‘‘Algorithmic System Planning of Air Handling Units’’, Project No. 22289 N/1, funded by the program for promoting the Industrial Collective Research (IGF) of the German Ministry of Economic Affairs and Climate Action (BMWK), approved by the Deutsches Zentrum für Luft- und Raumfahrt (DLR). We want to thank all the participants of the working group for the constructive collaboration.


## How to Cite
tbd.
