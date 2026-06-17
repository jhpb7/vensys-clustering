from copy import deepcopy
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .utils import compute_required_volume_flows
from .load_distributions import build_zone_pmfs


ONE_EPS = np.nextafter(1.0, 0.0)


def count_unique_rows(samples: pd.DataFrame | dict) -> pd.DataFrame:
    """
    Count identical sampled load cases and compute their relative frequencies.
    """
    df = pd.DataFrame(samples)

    row_counts = df.apply(tuple, axis=1).value_counts().reset_index(name="frequency")

    row_counts_expanded = pd.DataFrame(
        row_counts["index"].to_list(),
        columns=df.columns,
    )

    row_counts_expanded["frequency"] = row_counts["frequency"].to_numpy()
    row_counts_expanded["frequency"] /= row_counts_expanded["frequency"].sum()

    return row_counts_expanded


def sampled_load_cases_to_dict(row_counts_df: pd.DataFrame) -> dict:
    """
    Convert counted sampled load cases to the YAML-compatible scenario format.
    """
    result = {"load_cases": {}, "time_share": {}}

    for scenario_id, row in row_counts_df.iterrows():
        scenario_key = str(scenario_id)
        result["load_cases"][scenario_key] = {}
        for room_name in row.index:
            if room_name == "frequency":
                continue

            result["load_cases"][scenario_key][room_name] = float(row[room_name])

        result["time_share"][scenario_key] = float(row["frequency"])

    return result


def normalize_pmf(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize a probability mass function and return PMF and CDF.
    """
    pmf = np.asarray(pmf, dtype=float)

    pmf_sum = pmf.sum()
    if not np.isfinite(pmf_sum) or pmf_sum <= 0:
        raise ValueError("PMF must have a positive finite sum.")

    pmf = pmf / pmf_sum

    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0

    return pmf, cdf


def inverse_transform_discrete(
    u: np.ndarray,
    pmf: np.ndarray,
    demand_values: np.ndarray,
) -> np.ndarray:
    """
    Map samples from U(0, 1) to a discrete distribution.
    """
    _, cdf = normalize_pmf(pmf)

    u = np.asarray(u, dtype=float)
    u = np.minimum(u, ONE_EPS)

    idx = np.searchsorted(cdf, u, side="right")
    idx = np.minimum(idx, len(cdf) - 1)

    return np.asarray(demand_values, dtype=float)[idx]


def sample_zone_pmfs_sobol(
    zone_pmfs_by_time: dict,
    num_samples: int = 1024,
    hours: Iterable | None = None,
    scramble: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Generate sampled load cases from zone PMFs using a Sobol sequence.

    Returns
    -------
    room_to_samples:
        Mapping room -> array with shape (num_samples, num_hours, 1)

    hours:
        Array of sampled time slots.
    """
    if hours is None:
        hours = sorted(zone_pmfs_by_time.keys())
    else:
        hours = list(hours)

    rooms = sorted(
        {room for hour in hours for room in zone_pmfs_by_time.get(hour, {}).keys()}
    )

    num_hours = len(hours)
    num_rooms = len(rooms)
    sobol_dimension = num_hours * num_rooms

    if sobol_dimension == 0:
        return {}, np.array(hours)

    sobol = qmc.Sobol(d=sobol_dimension, scramble=scramble)
    unit_samples = sobol.random(num_samples)

    column_index = {}
    k = 0

    for hour_idx, hour in enumerate(hours):
        time_data = zone_pmfs_by_time.get(hour, {})

        for room in rooms:
            if room in time_data:
                column_index[(room, hour_idx)] = k
                k += 1

    room_to_samples = {}

    for room in rooms:
        arr = np.full((num_samples, num_hours), np.nan, dtype=float)

        for hour_idx, hour in enumerate(hours):
            key = (room, hour_idx)

            if key not in column_index:
                continue

            pmf, demand_values = zone_pmfs_by_time[hour][room]
            sample_column = unit_samples[:, column_index[key]]

            arr[:, hour_idx] = inverse_transform_discrete(
                sample_column,
                pmf,
                demand_values,
            )

        room_to_samples[room] = arr[..., None]

    return room_to_samples, np.array(hours)


def create_sampled_load_cases(
    general_data: dict,
    raw_load_case_data: dict,
    num_samples: int = 1024,
    resolution: float = 1.0,
    scramble: bool = True,
    include_revision: bool = False,
) -> dict:
    """
    Create sampled load cases from binomial room-load distributions.

    This function computes the room-level distributions, samples them with
    a Sobol sequence, counts identical sampled load cases, and returns the
    standard scenario/time_share dictionary.
    """
    rooms_to_merge = deepcopy(raw_load_case_data["rooms_to_merge"])

    df = compute_required_volume_flows(
        general_data,
        raw_load_case_data,
        overview_flag=True,
        include_revision=include_revision,
    )

    zone_pmfs_by_time = build_zone_pmfs(
        df=df,
        rooms_to_merge=rooms_to_merge,
        resolution=resolution,
    )

    room_samples, _ = sample_zone_pmfs_sobol(
        zone_pmfs_by_time=zone_pmfs_by_time,
        num_samples=num_samples,
        scramble=scramble,
    )

    flat_room_samples = {room: values.ravel() for room, values in room_samples.items()}

    row_counts_df = count_unique_rows(flat_room_samples)

    row_counts_df = row_counts_df.loc[
        row_counts_df.drop(columns="frequency").sum(axis=1).sort_values().index
    ].reset_index(drop=True)

    return sampled_load_cases_to_dict(row_counts_df)
