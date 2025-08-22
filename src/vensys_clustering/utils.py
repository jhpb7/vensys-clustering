from typing import Dict, List, Any, Union, Optional, Tuple
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ruamel.yaml import YAML
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Type Aliases
# ─────────────────────────────────────────────────────────────────────────────
RoomID = str
TimeSlot = str
RoomType = str

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOM = "room"
TIME_SLOT = "time_slot"
Q_REQUIRED = "q_required"


# ─────────────────────────────────────────────────────────────────────────────
# YAML Loader
# ─────────────────────────────────────────────────────────────────────────────
def get_yaml_loader() -> YAML:
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    return yaml


def load_yaml(filename: str) -> Dict[str, Any]:
    """
    Load YAML data from a file.

    Parameters:
    -----------
    filename : str
        Path to the YAML file.

    Returns:
    --------
    Dict[str, Any]
        Parsed YAML content as a dictionary.
    """
    with open(filename, "r") as f:
        data = get_yaml_loader().load(f)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_required_volume_flows(
    standard_data: Dict[str, Any],
    building_data: Dict[str, Any],
    overview_flag: bool = False,
    include_revision=False,
) -> pd.DataFrame:
    """
    Compute required volume flows per room and time slot.

    Parameters:
    -----------
    standard_data : Dict[str, Any]
        Standard values for occupancy and ventilation rates.
    building_data : Dict[str, Any]
        Room-specific parameters including room types, areas, and max persons.
    overview_flag : bool
        Whether to return full overview dataframe or just the key outputs.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing required volume flows.
    """
    rooms: List[RoomID] = building_data["rooms"]
    roomtype_map: Dict[RoomID, RoomType] = building_data["roomtype"]
    area_map: Dict[RoomID, float] = building_data["area"]
    max_person_map: Dict[RoomID, int] = building_data["max_num_person"]
    if include_revision:
        q_rev_tot: Dict[RoomID, float] = building_data["q_revision_tot"]

    occupancy_data: Dict[RoomType, Dict[TimeSlot, float]] = standard_data[
        "occupancy_data"
    ]
    q_per_person: Dict[RoomType, float] = standard_data["q_per_person"]["values"]
    q_per_area: Dict[RoomType, float] = standard_data["q_per_area"]["values"]

    rows: List[Dict[str, Union[str, float, int]]] = []

    for room in rooms:
        rtype = roomtype_map[room]
        area = area_map[room]
        max_persons = max_person_map[room]
        occupancy = occupancy_data[rtype]
        q_pp = q_per_person[rtype]
        q_pa = q_per_area[rtype]
        if include_revision:
            q_rev_tot_r = q_rev_tot[room]

        for time, occ in occupancy.items():
            rows.append(
                {
                    ROOM: room,
                    "room_type": rtype,
                    "area": area,
                    "max_persons": max_persons,
                    TIME_SLOT: time,
                    "occupancy": occ,
                    "q_per_person": q_pp,
                    "q_per_area": q_pa,
                }
            )
            if include_revision:
                rows[-1]["q_revision_tot"] = q_rev_tot_r

    df: pd.DataFrame = pd.DataFrame(rows)

    if include_revision:
        df = df.assign(
            q_persons=df["max_persons"] * df["occupancy"] * df["q_per_person"],
            q_area=df["area"] * df["q_per_area"],
            q_revision=df["occupancy"] * df["q_revision_tot"],
        )
        df[Q_REQUIRED] = df[["q_persons", "q_area", "q_revision"]].max(axis=1)

    else:
        df = df.assign(
            q_persons=df["max_persons"] * df["occupancy"] * df["q_per_person"],
            q_area=df["area"] * df["q_per_area"],
        )
        df[Q_REQUIRED] = df[["q_persons", "q_area"]].max(axis=1)

    return (
        df
        if overview_flag
        else df[[ROOM, TIME_SLOT, Q_REQUIRED]].sort_values([ROOM, TIME_SLOT])
    )


def merge_rooms(df: pd.DataFrame, building_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Merge rooms based on specified grouping and sum q_required.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'room', 'time_slot', and 'q_required'.
    building_data : Dict[str, Any]
        Contains information on which rooms to merge.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with merged rooms and summed q_required values.
    """
    rooms_to_merge: Dict[RoomID, List[RoomID]] = building_data["rooms_to_merge"]

    df_merged = df.copy()
    rooms_to_exclude = pd.Series(rooms_to_merge.values()).explode().tolist()
    df_merged = df_merged[~df_merged[ROOM].isin(rooms_to_exclude)]

    for new_room, group in rooms_to_merge.items():
        merged_rows = (
            df[df[ROOM].isin(group)].groupby(TIME_SLOT)[Q_REQUIRED].sum().reset_index()
        )
        merged_rows[ROOM] = new_room
        df_merged = pd.concat([df_merged, merged_rows], ignore_index=True)

    return df_merged[[ROOM, TIME_SLOT, Q_REQUIRED]].sort_values([ROOM, TIME_SLOT])


def cluster_time_slots_by_q(
    df: pd.DataFrame, n_clusters: int = 3
) -> Dict[RoomID, List[float]]:
    """
    Cluster time slots based on q_required profiles across rooms using K-Means.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame with columns ['room', 'time_slot', 'q_required'].

    n_clusters : int
        The number of clusters to group time slots into.

    Returns:
    --------
    Dict[str, List[float]]
        A dictionary mapping each room to a list of average q_required values
        per time cluster. Format: { room_id: [avg_q_cluster_0, ..., avg_q_cluster_n] }
    """
    pivot_df = df.pivot(index=TIME_SLOT, columns=ROOM, values=Q_REQUIRED).fillna(0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    time_slot_clusters = kmeans.fit_predict(pivot_df)

    time_cluster_map = pd.DataFrame(
        {TIME_SLOT: pivot_df.index, "time_cluster": time_slot_clusters}
    )

    df_clustered = df.merge(time_cluster_map, on=TIME_SLOT)

    room_cluster_q = (
        df_clustered.groupby([ROOM, "time_cluster"])[Q_REQUIRED]
        .mean()
        .unstack(fill_value=0)
    )

    unique, counts = np.unique(time_slot_clusters, return_counts=True)
    time_shares = dict(
        zip([int(x) for x in unique], [float(x / sum(counts)) for x in counts])
    )

    clustered_q_dict: Dict[RoomID, List[float]] = room_cluster_q.to_dict()

    return clustered_q_dict, time_shares


def analyze_cluster_quality(
    df: pd.DataFrame, elbow_max_k: int = 10
) -> Dict[str, Dict[int, float]]:
    """
    Analyze clustering quality using inertia (elbow method) and silhouette score.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns ['room', 'time_slot', 'q_required'].
    elbow_max_k : int
        Maximum number of clusters to test.

    Returns:
    --------
    Dict[str, Dict[int, float]]
        A dictionary with:
            - 'inertia': { k: inertia value }
            - 'silhouette': { k: silhouette score }
    """
    pivot_df = df.pivot(index=TIME_SLOT, columns=ROOM, values=Q_REQUIRED).fillna(0)

    inertia: Dict[int, float] = {}
    silhouette: Dict[int, float] = {}

    for k in range(2, elbow_max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pivot_df)
        inertia[k] = kmeans.inertia_
        silhouette[k] = silhouette_score(pivot_df, labels)

    return {"inertia": inertia, "silhouette": silhouette}


def save_scenario_data_to_yaml(
    data: Dict[
        int | str,
        Dict[int | str, Union[float, Dict[str, float]]],
    ],
    filename: str,
    quantile: Optional[str] = None,
) -> None:
    """
    Convert scenario-room data to standard Python types and save to YAML.

    Parameters:
        data: Dictionary of scenario -> room -> stats (mean or {mean, q99}).
        filename: Path to save the output YAML file.
    """
    # Normalize all data to Python-native types
    result = {}
    load_case_data = data["load_cases"]
    for scenario, rooms in load_case_data.items():
        result[scenario] = {"room": {}}
        for room, val in rooms.items():
            if isinstance(val, dict):  # Already has mean and quantile
                if quantile not in val:
                    raise KeyError(f"Quantile {quantile} not in keys: {val.keys()}")
                result[scenario]["room"][room] = {
                    "mean": float(val["mean"]),
                    quantile: float(val[quantile]),
                }
            else:  # Only mean value
                result[scenario]["room"][room] = {"mean": float(val)}

    wrapped = {"scenario": result, "time_share": data["time_share"]}

    # Write YAML using ruamel.yaml
    yaml = YAML()
    yaml.default_flow_style = False
    with open(filename, "w") as f:
        yaml.dump(wrapped, f)


def load_scenario_data_from_yaml(
    filename: str,
) -> None:
    # Write YAML using ruamel.yaml
    yaml = YAML()
    yaml.default_flow_style = False
    with open(filename, "r") as f:
        return yaml.load(f)


def compute_theoretical_max_q_per_zone(
    standard_data: Dict[str, Any],
    building_data: Dict[str, Any],
    include_revision: bool = False,
) -> Dict[str, float]:
    """
    Compute the theoretical maximum q_required per zone.
    Rooms listed in rooms_to_merge are aggregated; others are treated as their own zones.

    Parameters:
    -----------
    standard_data : Dict[str, Any]
        Contains q_per_person and q_per_area per room type.
    building_data : Dict[str, Any]
        Contains room properties and optionally 'rooms_to_merge'.
    include_revision : bool
        If True, consider q_revision_tot.

    Returns:
    --------
    Dict[str, float]
        Zone name -> theoretical max q_required.
    """
    roomtype_map = building_data["roomtype"]
    area_map = building_data["area"]
    max_person_map = building_data["max_num_person"]
    rooms = building_data["rooms"]
    q_per_person = standard_data["q_per_person"]["values"]
    q_per_area = standard_data["q_per_area"]["values"]

    rooms_to_merge = building_data.get("rooms_to_merge", {})
    merged_rooms = set(r for group in rooms_to_merge.values() for r in group)

    if include_revision:
        q_rev_tot = building_data["q_revision_tot"]

    zone_max_q: Dict[str, float] = {}

    # 1. Handle merged rooms
    for zone, room_list in rooms_to_merge.items():
        total_q = 0.0
        for room in room_list:
            rtype = roomtype_map[room]
            area = area_map[room]
            max_persons = max_person_map[room]
            q_pp = q_per_person[rtype]
            q_pa = q_per_area[rtype]

            q_persons = max_persons * q_pp
            q_area = area * q_pa
            q_required = max(q_persons, q_area)

            if include_revision:
                q_rev = q_rev_tot[room]
                q_required = max(q_required, q_rev)

            total_q += q_required

        zone_max_q[zone] = total_q

    # 2. Handle unmerged rooms (as their own zones)
    for room in rooms:
        if room in merged_rooms:
            continue  # Already accounted for
        rtype = roomtype_map[room]
        area = area_map[room]
        max_persons = max_person_map[room]
        q_pp = q_per_person[rtype]
        q_pa = q_per_area[rtype]

        q_persons = max_persons * q_pp
        q_area = area * q_pa
        q_required = max(q_persons, q_area)

        if include_revision:
            q_rev = q_rev_tot[room]
            q_required = max(q_required, q_rev)

        zone_max_q[room] = q_required  # Room is its own zone

    return zone_max_q


def add_max_load_case(
    max_load_case: Dict[str, float],
    load_cases: Dict[str, Dict[str, float]],
    time_shares: Dict[int, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, float]]:
    """
    Adds a maximum theoretical load case to the existing load_cases and time_shares.

    Parameters:
    -----------
    max_values : Dict[str, float]
        Mapping of zone or room IDs to their maximum theoretical q_required values.

    load_cases : Dict[str, Dict[str, float]]
        Existing load cases structured as cluster name -> { room_id: value }.

    time_shares : Dict[int, float]
        Mapping of cluster indices to their time share (relative duration or frequency).
        A new entry with time share of 0 will be added for the max load case.


    Returns:
    --------
    Tuple[Dict[str, Dict[str, float]], Dict[int, float]]
        Updated load_cases including the 'max' case,
        and updated time_shares with an entry for the max case.
    """
    load_cases["max"] = max_load_case
    time_shares[max(time_shares.keys(), default=-1) + 1] = 0.0
    return load_cases, time_shares
