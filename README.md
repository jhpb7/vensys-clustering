# `VenSyS_Clustering`: Ventilation System Scenario Reduction Toolkit

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

```bash
# from the project root
pip install -e .

# or install only runtime dependencies
pip install numpy pandas scipy scikit-learn matplotlib ruamel.yaml
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

    Add zero_inflated_rooms if some rooms are often unoccupied (probabilistic method).

    The max load case ensures system sizing robustness.


## Contributing

Contributions are welcome!

    Put runnable demos in examples/.

    Add docstrings for new functions.

    Keep style consistent with existing code.


## How to Cite
tbd.