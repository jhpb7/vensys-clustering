import numpy as np
import pandas as pd
from scipy.stats import binom, wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter


def max_binom_volumeflow_pmf(
    n: int,
    p: float,
    q_building: float,
    q_per_person: float,
    resolution: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the PMF and support for the required volume flow in a room,
    accounting for uncertainty in occupancy.

    The demand is defined as:
        max(q_persons, q_building)
    where:
        - q_persons is Binomial(n, p) * q_per_person
        - q_building is a constant area-based requirement

    Parameters:
        n: int
            Maximum number of people in the room.
        p: float
            Probability that a person is present (occupancy rate).
        q_building: float
            Area-based constant ventilation demand for the room.
        q_per_person: float
            Ventilation demand per person.
        resolution: float
            Discretization resolution in volume flow units.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            - PMF array over volume flows
            - Corresponding support array (volume flow values)
    """

    x_people = np.arange(0, n + 1)
    binom_pmf_prime = binom.pmf(x_people, n, p)

    pmf_people = binom_pmf_prime

    q_values = x_people * q_per_person

    pmf_dict = defaultdict(float)
    for q, prob in zip(q_values, pmf_people):
        if q < q_building:
            key = round(q_building / resolution) * resolution
        else:
            key = round(q / resolution) * resolution
        pmf_dict[key] += prob

    support = np.array(sorted(pmf_dict.keys()))
    pmf = np.array([pmf_dict[q] for q in support])
    return pmf, support


def merge_pmfs(
    pmfs: List[np.ndarray], supports: List[np.ndarray], resolution: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve multiple PMFs and align their supports accordingly.

    Parameters:
        pmfs: List of PMF arrays to convolve.
        supports: Corresponding support arrays.
        resolution: Volume flow resolution.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            - Merged PMF
            - Merged support
    """
    merged = {float(v): float(p) for v, p in zip(supports[0], pmfs[0])}

    for pmf, support in zip(pmfs[1:], supports[1:]):
        new = defaultdict(float)

        for v1, p1 in merged.items():
            for v2, p2 in zip(support, pmf):
                key = round((v1 + float(v2)) / resolution) * resolution
                new[key] += p1 * float(p2)

        merged = dict(new)

    merged_support = np.array(sorted(merged.keys()), dtype=float)
    merged_pmf = np.array([merged[v] for v in merged_support], dtype=float)
    merged_pmf /= merged_pmf.sum()

    return merged_pmf, merged_support


def build_zone_pmfs(
    df: pd.DataFrame,
    rooms_to_merge: Dict[str, List[str]],
    resolution: float = 1.0,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Build PMFs per zone or room across all time slots.

    Parameters:
        df: DataFrame with room-level time slot data.
        rooms_to_merge: Mapping from zone name to list of room names to merge.
        resolution: Volume flow resolution.

    Returns:
        Dict[time_slot, Dict[zone_or_room, (pmf, support)]]
    """

    zone_pmfs_by_time = {}

    for time_slot in df["time_slot"].unique():
        df_t = df[df["time_slot"] == time_slot]
        zone_pmfs = {}
        merged_rooms = set()

        for zone, room_list in rooms_to_merge.items():
            pmfs, supports = [], []
            for room in room_list:
                row = df_t[df_t["room"] == room].iloc[0]
                n, p = int(row["max_persons"]), float(row["occupancy"])
                q_building, q_per_person = float(row["q_area"]), float(
                    row["q_per_person"]
                )
                pmf, support = max_binom_volumeflow_pmf(
                    n,
                    p,
                    q_building,
                    q_per_person,
                    resolution,
                )
                pmfs.append(pmf)
                supports.append(support)
                merged_rooms.add(room)
            merged_pmf, merged_support = merge_pmfs(pmfs, supports, resolution)
            zone_pmfs[zone] = (merged_pmf, merged_support)

        for _, row in df_t.iterrows():
            room = row["room"]
            if room in merged_rooms:
                continue
            n, p = int(row["max_persons"]), float(row["occupancy"])
            q_building, q_per_person = float(row["q_area"]), float(row["q_per_person"])
            pmf, support = max_binom_volumeflow_pmf(
                n, p, q_building, q_per_person, resolution
            )
            zone_pmfs[room] = (pmf, support)

        zone_pmfs_by_time[time_slot] = zone_pmfs

    return zone_pmfs_by_time


def compute_distance_matrix(
    zone_pmfs_by_time: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> pd.DataFrame:
    time_slots = list(zone_pmfs_by_time.keys())
    dist_matrix = np.zeros((len(time_slots), len(time_slots)))

    for i in range(len(time_slots)):
        for j in range(i + 1, len(time_slots)):
            t1, t2 = time_slots[i], time_slots[j]

            # Use only zones that exist in both time slots
            zones = set(zone_pmfs_by_time[t1]).intersection(zone_pmfs_by_time[t2])

            dists = []

            for zone in zones:
                pmf1, support1 = zone_pmfs_by_time[t1][zone]
                pmf2, support2 = zone_pmfs_by_time[t2][zone]

                pmf1 = np.asarray(pmf1, dtype=float)
                pmf2 = np.asarray(pmf2, dtype=float)
                support1 = np.asarray(support1, dtype=float)
                support2 = np.asarray(support2, dtype=float)

                # Skip invalid / empty distributions
                if (
                    len(pmf1) == 0
                    or len(pmf2) == 0
                    or len(pmf1) != len(support1)
                    or len(pmf2) != len(support2)
                    or not np.isfinite(pmf1).all()
                    or not np.isfinite(pmf2).all()
                    or pmf1.sum() <= 0
                    or pmf2.sum() <= 0
                ):
                    continue

                # Normalise to be safe
                pmf1 = pmf1 / pmf1.sum()
                pmf2 = pmf2 / pmf2.sum()

                dists.append(
                    wasserstein_distance(
                        support1,
                        support2,
                        u_weights=pmf1,
                        v_weights=pmf2,
                    )
                )

            if len(dists) == 0:
                dist = np.nan
            else:
                dist = np.mean(dists)

            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return pd.DataFrame(dist_matrix, index=time_slots, columns=time_slots)


def cluster_time_slots_from_distribution(
    df: pd.DataFrame,
    rooms_to_merge: Dict[str, List[str]],
    n_clusters: int = 3,
    resolution: float = 1.0,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Perform clustering on time slots based on distribution-aware demand vectors.

    Parameters:
        df: DataFrame with room data (including occupancy).
        rooms_to_merge: Mapping of zone names to room lists.
        n_clusters: Number of clusters to create.
        resolution: Volume flow resolution for PMFs.

    Returns:
        Tuple:
            - Mapping from time_slot to cluster label
            - PMF dictionary from build_zone_pmfs
    """

    zone_pmfs_by_time = build_zone_pmfs(df, rooms_to_merge, resolution)
    dist_matrix = compute_distance_matrix(zone_pmfs_by_time)

    model = AgglomerativeClustering(
        metric="precomputed", linkage="average", n_clusters=n_clusters
    )
    labels = model.fit_predict(dist_matrix.values)

    return dict(zip(dist_matrix.index, labels)), zone_pmfs_by_time


def extract_cluster_representatives(
    cluster_labels: Dict[str, int],
    zone_pmfs_by_time: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    quantile: float = 0.95,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Extract representative demand values (mean and quantile) per cluster and zone.

    Parameters:
        cluster_labels: Mapping from time slot to cluster index.
        zone_pmfs_by_time: PMFs and supports for each time slot and zone.
        quantile: Quantile level to extract (e.g. 0.95 for 95th percentile).

    Returns:
        Nested dict:
            cluster -> zone -> {"mean": ..., "q95": ...}
    """
    clusters = defaultdict(list)
    for time_slot, label in cluster_labels.items():
        clusters[label].append(time_slot)

    result = {}

    for label, time_slots in clusters.items():
        zone_flows = defaultdict(list)

        for time_slot in time_slots:
            for zone, (pmf, support) in zone_pmfs_by_time[time_slot].items():
                zone_flows[zone].append((pmf, support))

        result[label] = {}

        for zone, pmf_support_list in zone_flows.items():
            all_supports = [support for _, support in pmf_support_list]
            min_support = min(s[0] for s in all_supports)
            max_support = max(s[-1] for s in all_supports)
            resolution = all_supports[0][1] - all_supports[0][0]
            common_support = np.arange(
                min_support, max_support + resolution, resolution
            )

            summed_pmf = np.zeros_like(common_support)
            for pmf, support in pmf_support_list:
                aligned_pmf = np.zeros_like(common_support)
                for i, val in enumerate(support):
                    idx = int(round((val - min_support) / resolution))
                    aligned_pmf[idx] = pmf[i]
                summed_pmf += aligned_pmf

            avg_pmf = summed_pmf / len(pmf_support_list)
            mean_flow = np.sum(common_support * avg_pmf)
            cdf = np.cumsum(avg_pmf)
            quantile_flow = common_support[np.searchsorted(cdf, quantile)]

            result[label][zone] = {
                "mean": mean_flow,
                f"q{int(quantile * 100)}": quantile_flow,
            }

    return result


def evaluate_cluster_silhouette_scores(
    dist_matrix: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 10
) -> List[Tuple[int, float]]:
    """
    Evaluate silhouette scores for different cluster counts using agglomerative clustering.

    Parameters:
    -----------
    dist_matrix : pd.DataFrame
        Precomputed symmetric distance matrix between time slots.
    min_clusters : int
        Minimum number of clusters to test (inclusive).
    max_clusters : int
        Maximum number of clusters to test (inclusive).

    Returns:
    --------
    List[Tuple[int, float]]
        List of (n_clusters, silhouette_score) tuples.
    """
    scores = {}
    X = dist_matrix.values

    for k in range(min_clusters, max_clusters + 1):
        model = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="precomputed"
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="precomputed")
        scores[k] = score

    return scores


def compute_cluster_barycenter_pmfs(
    cluster_labels: Dict[str, int],
    zone_pmfs_by_time: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    resolution: float = 1.0,
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute Wasserstein barycenter PMFs for each cluster and zone/room.

    Parameters:
    -----------
    cluster_labels : Dict[str, int]
        Mapping from time slot to cluster label.
    zone_pmfs_by_time : Dict[str, Dict[str, Tuple[pmf, support]]]
        Original PMFs and supports per time slot and zone/room.
    resolution : float
        Resolution of the volume flow support.

    Returns:
    --------
    Dict[int, Dict[str, Tuple[pmf, support]]]
        Mapping from cluster label to zone to (barycenter_pmf, support).
    """

    def compute_1d_wasserstein_barycenter(pmfs: List[np.ndarray]) -> np.ndarray:
        """
        Compute the 1D Wasserstein barycenter from a list of aligned 1D PMFs.
        """
        n = len(pmfs[0])
        avg_cdf = np.zeros(n)
        for p in pmfs:
            avg_cdf += np.cumsum(p)
        avg_cdf /= len(pmfs)
        return np.diff(np.concatenate(([0], avg_cdf)))

    clusters = defaultdict(list)
    for time_slot, label in cluster_labels.items():
        clusters[label].append(time_slot)

    cluster_barycenters = {}

    for label, time_slots in clusters.items():
        zone_flows = defaultdict(list)

        for time_slot in time_slots:
            for zone, (pmf, support) in zone_pmfs_by_time[time_slot].items():
                zone_flows[zone].append((pmf, support))

        cluster_barycenters[label] = {}

        for zone, pmf_support_list in zone_flows.items():
            all_supports = [s for _, s in pmf_support_list]
            min_support = min(s[0] for s in all_supports)
            max_support = max(s[-1] for s in all_supports)
            common_support = np.arange(
                min_support, max_support + resolution, resolution
            )

            aligned_pmfs = []
            for pmf, support in pmf_support_list:
                aligned = np.zeros_like(common_support)
                for i, val in enumerate(support):
                    idx = int(round((val - min_support) / resolution))
                    aligned[idx] = pmf[i]
                aligned_pmfs.append(aligned)

            barycenter_pmf = compute_1d_wasserstein_barycenter(aligned_pmfs)
            cluster_barycenters[label][zone] = (barycenter_pmf, common_support)

    return cluster_barycenters


def summarize_barycenter_pmfs(
    cluster_barycenters: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    quantile: float = 0.95,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Compute summary statistics (mean and quantile) from barycenter PMFs per cluster and zone.

    Parameters:
    -----------
    cluster_barycenters : Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Mapping from cluster to zone to (barycenter_pmf, support).
    quantile : float
        Quantile level to compute (default: 0.95).

    Returns:
    --------
    Dict[int, Dict[str, Dict[str, float]]]
        Nested dictionary with mean and quantile per cluster and zone.
    """
    summary = {}

    for cluster, zone_data in cluster_barycenters.items():
        cluster = int(cluster)
        summary[cluster] = {}
        for zone, (pmf, support) in zone_data.items():
            mean = np.sum(pmf * support)
            cdf = np.cumsum(pmf)
            q_val = support[np.searchsorted(cdf, quantile)]
            summary[cluster][zone] = {"mean": mean, f"q{int(quantile * 100)}": q_val}

    return summary


def plot_zone_quantile_comparison(
    cluster_barycenters: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    zone_name: str | int,
    quantiles: List[float] = [0.05, 0.5, 0.95],
):
    """
    Plot low/mid/high quantiles for a specific zone across all clusters.

    Parameters:
    -----------
    cluster_barycenters : Dict[int, Dict[str, Tuple[pmf, support]]]
        Barycenter PMFs per cluster and zone.
    zone_name : str
        Name of the zone/room to compare across clusters.
    quantiles : List[float]
        List of quantiles to extract (e.g., [0.05, 0.5, 0.95]).
    """
    bars = []
    cluster_labels = []
    quantile_labels = [f"q{int(q*100)}" for q in quantiles]

    for cluster_id, zones in cluster_barycenters.items():
        if zone_name not in zones:
            continue
        pmf, support = zones[zone_name]
        cdf = np.cumsum(pmf)
        q_values = [support[np.searchsorted(cdf, q)] for q in quantiles]
        bars.append(q_values)
        cluster_labels.append(f"Cluster {cluster_id}")

    bars = np.array(bars)
    x = np.arange(len(cluster_labels))

    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(quantile_labels):
        ax.bar(x + i * width, bars[:, i], width, label=label)

    ax.set_ylabel("Volume Flow Demand")
    ax.set_title(f"Quantile Comparison for Zone '{zone_name}' Across Clusters")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cluster_labels)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def compute_cluster_time_shares(cluster_labels: Dict[str, int]) -> Dict[int, float]:
    """
    Compute the time share (relative frequency) of each cluster.

    Parameters:
        cluster_labels : Dict[time_slot, cluster_label]

    Returns:
        Dict[cluster_label, time_share]
    """
    counts = Counter(cluster_labels.values())
    total = sum(counts.values())
    return {int(cluster): count / total for cluster, count in counts.items()}
