from collections import defaultdict
from typing import Any, TypedDict


class DataEntry(TypedDict):
    params: dict[str, Any]
    values: dict[str, Any]


class GroupedEntry(TypedDict):
    params: dict[str, Any]
    values: list[dict[str, Any]]


def make_hashable(val: Any) -> Any:
    if isinstance(val, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
    elif isinstance(val, (list, tuple)):
        return tuple(make_hashable(v) for v in val)
    elif isinstance(val, set):
        return frozenset(make_hashable(v) for v in val)
    return val


def group_entries(entries: list[DataEntry]) -> list[GroupedEntry]:
    grouped_map = defaultdict(list)
    params_reference = {}

    for entry in entries:
        p_dict = entry["params"]
        v_dict = entry["values"]

        params_key = make_hashable(p_dict)

        grouped_map[params_key].append(v_dict)

        if params_key not in params_reference:
            params_reference[params_key] = p_dict

    unique_keys = list(grouped_map.keys())

    for i in range(len(unique_keys)):
        for j in range(len(unique_keys)):
            if i == j:
                continue

            set_i = set(unique_keys[i])
            set_j = set(unique_keys[j])

            if set_i < set_j:
                raise ValueError(
                    f"Ambiguity detected: Params {params_reference[unique_keys[i]]} "
                    f"are a subset of {params_reference[unique_keys[j]]}."
                )

    results: list[GroupedEntry] = []
    for key, values_list in grouped_map.items():
        results.append({"params": params_reference[key], "values": values_list})

    return results
