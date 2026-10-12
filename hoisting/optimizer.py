from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .domain import (
    Component,
    TYPE_SPECS,
    component_indegrees,
    component_successors,
    decode_priority_vector,
    estimate_wind_risk,
)


ProgressCallback = Callable[[int, int, np.ndarray], None]


@dataclass
class OptimizationResult:
    components: list[Component]
    pareto_priorities: list[np.ndarray]
    pareto_objectives: np.ndarray
    history: dict[str, list[float]]
    plans: dict[str, dict]
    cranes: list[dict]
    metadata: dict

    @property
    def recommended_plan(self) -> dict:
        return self.plans["综合推荐"]


def _gantry_specs(
    cranes: Sequence[dict],
    components: Sequence[Component],
) -> dict[str, np.ndarray]:
    minimum_x = min(component.x - component.length_m * 0.5 for component in components)
    maximum_x = max(component.x + component.length_m * 0.5 for component in components)
    minimum_y = min(component.y - component.width_m * 0.5 for component in components)
    maximum_y = max(component.y + component.width_m * 0.5 for component in components)
    maximum_z = max(component.z + component.height_m * 0.5 for component in components)
    return {
        "capacities": np.asarray([crane["capacity_t"] for crane in cranes], dtype=float),
        "rates": np.asarray([crane["rate_yuan_h"] for crane in cranes], dtype=float),
        "rail_start_x": np.asarray(
            [crane.get("rail_start_x", minimum_x - 8.0) for crane in cranes],
            dtype=float,
        ),
        "rail_end_x": np.asarray(
            [crane.get("rail_end_x", maximum_x + 8.0) for crane in cranes],
            dtype=float,
        ),
        "trolley_min_y": np.asarray(
            [crane.get("trolley_min_y", minimum_y - 2.0) for crane in cranes],
            dtype=float,
        ),
        "trolley_max_y": np.asarray(
            [crane.get("trolley_max_y", maximum_y + 2.0) for crane in cranes],
            dtype=float,
        ),
        "max_lift_height": np.asarray(
            [crane.get("max_lift_height", maximum_z + 6.0) for crane in cranes],
            dtype=float,
        ),
        "long_speed": np.asarray(
            [crane.get("long_speed_m_min", 18.0) for crane in cranes],
            dtype=float,
        ),
        "trolley_speed": np.asarray(
            [crane.get("trolley_speed_m_min", 12.0) for crane in cranes],
            dtype=float,
        ),
        "hoist_speed": np.asarray(
            [crane.get("hoist_speed_m_min", 8.0) for crane in cranes],
            dtype=float,
        ),
        "minimum_spacing": np.asarray(
            [crane.get("minimum_spacing_m", 8.0) for crane in cranes],
            dtype=float,
        ),
    }


def schedule_sequence(
    sequence: Sequence[int],
    components: Sequence[Component],
    cranes: Sequence[dict],
    setup_minutes: float = 8.0,
    travel_speed_m_min: float = 18.0,
    prefer_home_zone: bool = True,
) -> dict:
    gantry = _gantry_specs(cranes, components)
    capacities = gantry["capacities"]
    rates = gantry["rates"]
    crane_available = np.zeros(len(cranes), dtype=float)
    crane_current_x = np.asarray(
        [
            crane.get(
                "initial_x",
                (crane["position"][0] if "position" in crane else 0.0),
            )
            for crane in cranes
        ],
        dtype=float,
    )
    crane_current_y = np.asarray(
        [
            crane.get(
                "initial_y",
                (crane["position"][1] if "position" in crane else 0.0),
            )
            for crane in cranes
        ],
        dtype=float,
    )
    home_x = crane_current_x.copy()
    home_order = np.argsort(home_x)
    sorted_home_x = home_x[home_order]
    home_boundaries = (
        (sorted_home_x[:-1] + sorted_home_x[1:]) * 0.5
        if len(sorted_home_x) > 1
        else np.asarray([], dtype=float)
    )
    corridor_width = float(
        np.max(gantry["minimum_spacing"]) if len(cranes) > 1 else 0.0
    )
    corridor_available_until = 0.0
    crane_active_minutes = np.zeros(len(cranes), dtype=float)
    crane_movement_minutes = np.zeros(len(cranes), dtype=float)
    crane_move_count = np.zeros(len(cranes), dtype=int)
    rows: list[dict] = []
    unassigned: list[str] = []
    component_risk_factors: dict[int, dict[str, float]] = {}
    row_by_component_index: dict[int, dict] = {}

    for order, component_index in enumerate(sequence, start=1):
        component = components[component_index]
        eligible = np.where(
            (capacities >= component.weight_t)
            & (component.x >= gantry["rail_start_x"])
            & (component.x <= gantry["rail_end_x"])
            & (component.y >= gantry["trolley_min_y"])
            & (component.y <= gantry["trolley_max_y"])
            & (
                component.z + component.height_m * 0.5
                <= gantry["max_lift_height"]
            )
        )[0]
        preferred_crane = int(np.argmin(np.abs(home_x - component.x)))
        in_corridor = bool(
            len(home_boundaries)
            and np.any(
                np.abs(home_boundaries - component.x)
                <= corridor_width * 0.5
            )
        )
        if prefer_home_zone:
            if preferred_crane in eligible:
                # Keep multi-crane work partitioned by home side so cranes
                # approach the station center instead of sweeping the full site.
                eligible = np.asarray([preferred_crane], dtype=int)
        if len(eligible) == 0:
            unassigned.append(component.id)
            continue
        long_travel = np.abs(component.x - crane_current_x)
        trolley_travel = np.abs(component.y - crane_current_y)
        hoist_height = max(component.z + component.height_m * 0.5, 0.5)
        hoist_travel = hoist_height / np.maximum(gantry["hoist_speed"], 0.1) * 1.75
        movement_minutes = (
            long_travel / np.maximum(gantry["long_speed"], 0.1)
            + trolley_travel / np.maximum(gantry["trolley_speed"], 0.1)
            + hoist_travel
        )
        allocatable_available = crane_available.copy()
        if (
            prefer_home_zone
            and in_corridor
            and preferred_crane in eligible
        ):
            allocatable_available[preferred_crane] = max(
                allocatable_available[preferred_crane],
                corridor_available_until,
            )
        finish = (
            allocatable_available
            + setup_minutes
            + movement_minutes
            + component.duration_min
        )
        finish[~np.isin(np.arange(len(cranes)), eligible)] = np.inf
        candidate_start = (
            allocatable_available + setup_minutes + movement_minutes
        )
        for candidate in range(len(cranes)):
            for other_index in range(len(cranes)):
                if candidate == other_index:
                    continue
                if (
                    abs(component.x - crane_current_x[other_index])
                    < gantry["minimum_spacing"][candidate]
                    and crane_available[other_index]
                    > candidate_start[candidate]
                ):
                    finish[candidate] += (
                        gantry["minimum_spacing"][candidate]
                        / max(gantry["long_speed"][candidate], 0.1)
                    )
        crane_index = int(np.argmin(finish))
        movement_minutes_current = float(movement_minutes[crane_index])
        start = (
            allocatable_available[crane_index]
            + setup_minutes
            + movement_minutes_current
        )
        end = start + component.duration_min
        if prefer_home_zone and in_corridor:
            corridor_available_until = end
        capacity_t = float(capacities[crane_index])
        capacity_utilization = component.weight_t / max(capacity_t, 0.1)
        capacity_limit_risk = float(
            np.clip((capacity_utilization - 0.55) / 0.40, 0.0, 1.0)
        )
        trolley_center_y = (
            gantry["trolley_min_y"][crane_index]
            + gantry["trolley_max_y"][crane_index]
        ) * 0.5
        trolley_half_span = max(
            (
                gantry["trolley_max_y"][crane_index]
                - gantry["trolley_min_y"][crane_index]
            )
            * 0.5,
            0.1,
        )
        trolley_offset_m = abs(component.y - trolley_center_y)
        trolley_utilization = trolley_offset_m / trolley_half_span
        trolley_risk = float(
            np.clip((trolley_utilization - 0.45) / 0.50, 0.0, 1.0)
        )
        lift_height = component.z + component.height_m * 0.5
        lift_height_utilization = lift_height / max(
            gantry["max_lift_height"][crane_index],
            0.1,
        )
        lift_height_limit_risk = float(
            np.clip((lift_height_utilization - 0.55) / 0.40, 0.0, 1.0)
        )
        lifting_risk = max(
            capacity_limit_risk,
            trolley_risk,
        )
        height_risk = lift_height_limit_risk
        obstacle_risk = float(np.clip(component.obstacle_level / 3.0, 0.0, 1.0))
        visibility_risk = float(np.clip(component.visibility_level / 2.0, 0.0, 1.0))
        ground_risk = float(np.clip(component.ground_level / 2.0, 0.0, 1.0))
        wind_risk = estimate_wind_risk(
            length_m=component.length_m,
            width_m=component.width_m,
            height_m=component.height_m,
            weight_t=component.weight_t,
        )
        rotation_risk = 1.0 if component.rotation_required else 0.0
        component_risk_factors[component_index] = {
            "capacity": lifting_risk,
            "height": height_risk,
            "obstacle": obstacle_risk,
            "visibility": visibility_risk,
            "ground": ground_risk,
            "wind": wind_risk,
            "rotation": rotation_risk,
        }
        crane_available[crane_index] = end
        crane_current_x[crane_index] = component.x
        crane_current_y[crane_index] = component.y
        crane_active_minutes[crane_index] += component.duration_min
        crane_movement_minutes[crane_index] += movement_minutes_current
        crane_move_count[crane_index] += 1
        row = {
            "order": order,
            "component_index": component_index,
            "component_id": component.id,
            "component_name": component.name,
            "component_type": (
                component.semantic_type or TYPE_SPECS[component.type]["label"]
            ),
            "start": start,
            "end": end,
            "duration": component.duration_min,
            "crane": cranes[crane_index]["id"],
            "crane_index": crane_index,
            "weight_t": component.weight_t,
            "x": component.x,
            "y": component.y,
            "z": component.z,
            "risk": 0.0,
            "capacity_t": capacity_t,
            "capacity_utilization": capacity_utilization,
            "long_travel_m": float(long_travel[crane_index]),
            "trolley_travel_m": float(trolley_travel[crane_index]),
            "hoist_travel_m": float(hoist_height),
            "movement_minutes": movement_minutes_current,
            "trolley_offset_m": float(trolley_offset_m),
            "trolley_utilization": float(trolley_utilization),
            "lift_height_utilization": float(lift_height_utilization),
        }
        rows.append(row)
        row_by_component_index[component_index] = row

    makespan = float(np.max(crane_available)) if len(crane_available) else 0.0
    crane_cost = float(np.sum(rates * crane_available / 60.0))
    move_cost = float(np.sum(crane_move_count) * 42.0)
    heavy_premium = float(
        sum(
            max(0.0, components[item].weight_t - 14.0) * 26.0
            for item in sequence
        )
    )
    total_cost = crane_cost + move_cost + heavy_premium

    adjacent_crossings = 0
    vertical_crossings = 0
    for previous, current in zip(sequence, sequence[1:]):
        first = components[previous]
        second = components[current]
        x_gap = abs(first.x - second.x)
        y_gap = abs(first.y - second.y)
        z_gap = abs(first.z - second.z)
        if x_gap < 5.0 and y_gap < 5.5 and z_gap > 2.4:
            adjacent_crossings += 1
        if first.z - second.z > 1.6:
            vertical_crossings += 1

    pair_count = max(len(sequence) - 1, 1)
    collision_risk = float(
        np.clip(
            0.75 * (adjacent_crossings / pair_count)
            + 0.25 * (vertical_crossings / pair_count),
            0.0,
            1.0,
        )
    )
    risk_weights = {
        "capacity": 0.20,
        "height": 0.15,
        "collision": 0.17,
        "obstacle": 0.13,
        "visibility": 0.10,
        "ground": 0.10,
        "wind": 0.10,
        "rotation": 0.05,
    }
    for position, component_index in enumerate(sequence):
        if component_index not in component_risk_factors:
            continue
        previous_index = sequence[position - 1] if position > 0 else None
        next_index = sequence[position + 1] if position + 1 < len(sequence) else None
        local_crossings = 0
        local_neighbors = 0
        for neighbor_index in (previous_index, next_index):
            if neighbor_index is None:
                continue
            local_neighbors += 1
            first = components[component_index]
            second = components[neighbor_index]
            if (
                abs(first.x - second.x) < 5.0
                and abs(first.y - second.y) < 5.5
                and abs(first.z - second.z) > 2.4
            ):
                local_crossings += 1
            if first.z - second.z > 1.6:
                local_crossings += 0.5
        local_collision_risk = float(
            np.clip(local_crossings / max(local_neighbors, 1), 0.0, 1.0)
        )
        factors = component_risk_factors[component_index]
        contributions = {
            "capacity": factors["capacity"] * risk_weights["capacity"] * 100.0,
            "height": factors["height"] * risk_weights["height"] * 100.0,
            "collision": local_collision_risk * risk_weights["collision"] * 100.0,
            "obstacle": factors["obstacle"] * risk_weights["obstacle"] * 100.0,
            "visibility": factors["visibility"] * risk_weights["visibility"] * 100.0,
            "ground": factors["ground"] * risk_weights["ground"] * 100.0,
            "wind": factors["wind"] * risk_weights["wind"] * 100.0,
            "rotation": factors["rotation"] * risk_weights["rotation"] * 100.0,
        }
        row_by_component_index[component_index]["risk"] = float(sum(contributions.values()))
        row_by_component_index[component_index]["risk_breakdown"] = contributions

    average_factors = {}
    for key in (
        "capacity",
        "height",
        "obstacle",
        "visibility",
        "ground",
        "wind",
        "rotation",
    ):
        values = [
            component_risk_factors[component_index][key]
            for component_index in sequence
            if component_index in component_risk_factors
        ]
        average_factors[key] = float(np.mean(values)) if values else 0.0
    risk_breakdown = {
        "起重能力与小车偏载": (
            average_factors["capacity"] * risk_weights["capacity"] * 100.0
        ),
        "起升高度": average_factors["height"] * risk_weights["height"] * 100.0,
        "空间碰撞": collision_risk * risk_weights["collision"] * 100.0,
        "障碍条件": average_factors["obstacle"] * risk_weights["obstacle"] * 100.0,
        "作业可视性": average_factors["visibility"] * risk_weights["visibility"] * 100.0,
        "地基条件": average_factors["ground"] * risk_weights["ground"] * 100.0,
        "风荷载": average_factors["wind"] * risk_weights["wind"] * 100.0,
        "翻身旋转": average_factors["rotation"] * risk_weights["rotation"] * 100.0,
    }
    risk_score = float(np.clip(sum(risk_breakdown.values()), 0.0, 100.0))
    spatial_risk_score = float(
        np.clip(risk_breakdown["空间碰撞"] / risk_weights["collision"], 0.0, 100.0)
    )
    base_environment_risk = (
        risk_breakdown["起升高度"]
        + risk_breakdown["障碍条件"]
        + risk_breakdown["作业可视性"]
        + risk_breakdown["地基条件"]
        + risk_breakdown["风荷载"]
        + risk_breakdown["翻身旋转"]
    ) / (
        risk_weights["height"]
        + risk_weights["obstacle"]
        + risk_weights["visibility"]
        + risk_weights["ground"]
        + risk_weights["wind"]
        + risk_weights["rotation"]
    )
    movement_ratio_values = [
        row["movement_minutes"]
        / max(row["movement_minutes"] + row["duration"], 0.01)
        for row in rows
    ]
    movement_stress = (
        float(np.mean(movement_ratio_values)) * 100.0
        if movement_ratio_values
        else 0.0
    )
    environment_risk_score = float(
        np.clip(
            base_environment_risk * 0.75 + movement_stress * 0.25,
            0.0,
            100.0,
        )
    )
    occupied_minutes = crane_active_minutes + crane_movement_minutes
    if len(occupied_minutes) > 1 and float(np.mean(occupied_minutes)) > 0:
        workload_cv = float(
            np.std(occupied_minutes) / np.mean(occupied_minutes)
        )
        equipment_balance_score = float(
            np.clip(workload_cv / 0.50, 0.0, 1.0) * 100.0
        )
    else:
        equipment_balance_score = 0.0

    return {
        "sequence": list(sequence),
        "schedule": rows,
        "makespan_min": makespan,
        "cost_yuan": total_cost,
        "risk_score": risk_score,
        "expert_objective": [
            makespan,
            total_cost,
            spatial_risk_score,
            environment_risk_score,
            equipment_balance_score,
        ],
        "spatial_risk_score": spatial_risk_score,
        "environment_risk_score": environment_risk_score,
        "equipment_balance_score": equipment_balance_score,
        "risk_breakdown": risk_breakdown,
        "cost_breakdown": {
            "龙门吊台班": crane_cost,
            "大车行走与小车定位": move_cost,
            "重载附加": heavy_premium,
        },
        "crane_workload": [
            {
                "crane": crane["id"],
                "equipment_type": crane.get("type", "龙门吊"),
                "active_minutes": float(crane_active_minutes[index]),
                "movement_minutes": float(crane_movement_minutes[index]),
                "occupied_minutes": float(occupied_minutes[index]),
                "moves": int(crane_move_count[index]),
            }
            for index, crane in enumerate(cranes)
        ],
        "unassigned": unassigned,
        "adjacent_crossings": adjacent_crossings,
        "crane_policy": (
            "home_zone" if prefer_home_zone else "earliest_finish"
        ),
    }


def _initial_population(
    population_size: int,
    components: Sequence[Component],
    successors: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    size = len(components)
    population: list[np.ndarray] = []
    layer_order = np.asarray([TYPE_SPECS[component.type]["layer"] for component in components])
    mileage = np.asarray([component.x for component in components])
    risk = np.asarray([component.risk for component in components])

    baseline = layer_order * 10.0 + mileage / max(float(np.max(mileage)), 1.0)
    population.append(np.asarray(baseline, dtype=float))
    population.append(np.asarray(-mileage + layer_order * 2.0, dtype=float))
    population.append(np.asarray(-risk + layer_order * 3.0, dtype=float))

    while len(population) < population_size:
        priority = (
            layer_order * rng.normal(6.5, 1.25)
            + mileage / max(float(np.max(mileage)), 1.0) * rng.normal(1.0, 0.45)
            + rng.normal(0.0, 1.0, size=size)
        )
        decode_priority_vector(priority, components, successors)
        population.append(priority)
    return population[:population_size]


def _fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    population_size = len(objectives)
    dominated_solutions: list[list[int]] = [[] for _ in range(population_size)]
    domination_count = np.zeros(population_size, dtype=int)
    fronts: list[list[int]] = [[]]

    for first in range(population_size):
        for second in range(first + 1, population_size):
            first_values = objectives[first]
            second_values = objectives[second]
            first_dominates = np.all(first_values <= second_values) and np.any(
                first_values < second_values
            )
            second_dominates = np.all(second_values <= first_values) and np.any(
                second_values < first_values
            )
            if first_dominates:
                dominated_solutions[first].append(second)
                domination_count[second] += 1
            elif second_dominates:
                dominated_solutions[second].append(first)
                domination_count[first] += 1

    for index, count in enumerate(domination_count):
        if count == 0:
            fronts[0].append(index)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for first in fronts[front_index]:
            for second in dominated_solutions[first]:
                domination_count[second] -= 1
                if domination_count[second] == 0:
                    next_front.append(second)
        front_index += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _crowding_distance(objectives: np.ndarray, front: Sequence[int]) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for objective_index in range(objectives.shape[1]):
        ordered = sorted(front, key=lambda index: objectives[index, objective_index])
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        minimum = objectives[ordered[0], objective_index]
        maximum = objectives[ordered[-1], objective_index]
        if maximum <= minimum:
            continue
        scale = maximum - minimum
        for position in range(1, len(ordered) - 1):
            previous_value = objectives[ordered[position - 1], objective_index]
            next_value = objectives[ordered[position + 1], objective_index]
            distance[ordered[position]] += (next_value - previous_value) / scale
    return distance


def _select_next_generation(
    priorities: list[np.ndarray],
    objectives: np.ndarray,
    population_size: int,
) -> tuple[list[np.ndarray], np.ndarray, list[list[int]]]:
    fronts = _fast_non_dominated_sort(objectives)
    selected_indices: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
        else:
            distance = _crowding_distance(objectives, front)
            sorted_front = sorted(front, key=lambda index: distance[index], reverse=True)
            selected_indices.extend(sorted_front[: population_size - len(selected_indices)])
            break
    return (
        [priorities[index].copy() for index in selected_indices],
        objectives[selected_indices].copy(),
        _fast_non_dominated_sort(objectives[selected_indices]),
    )


def _generate_reference_points(
    objective_count: int,
    divisions: int,
) -> np.ndarray:
    points: list[list[float]] = []

    def recurse(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            points.append([float(value) for value in prefix + [remaining]])
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, slots - 1)

    recurse([], divisions, objective_count)
    return np.asarray(points, dtype=float) / max(divisions, 1)


def _associate_reference_points(
    normalized_objectives: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    associations: list[int] = []
    distances: list[float] = []
    for objective in normalized_objectives:
        differences = reference_points - objective
        distance_squared = np.sum(differences * differences, axis=1)
        nearest = int(np.argmin(distance_squared))
        associations.append(nearest)
        distances.append(float(np.sqrt(distance_squared[nearest])))
    return np.asarray(associations, dtype=int), np.asarray(distances, dtype=float)


def _select_nsga3(
    priorities: list[np.ndarray],
    objectives: np.ndarray,
    population_size: int,
    reference_points: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, list[list[int]]]:
    fronts = _fast_non_dominated_sort(objectives)
    selected_indices: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
            continue

        needed = population_size - len(selected_indices)
        minimum = objectives.min(axis=0)
        maximum = objectives.max(axis=0)
        span = np.where(maximum > minimum, maximum - minimum, 1.0)
        normalized = (objectives - minimum) / span
        associations, distances = _associate_reference_points(
            normalized,
            reference_points,
        )
        niche_counts = np.zeros(len(reference_points), dtype=int)
        if selected_indices:
            selected_associations, _ = _associate_reference_points(
                normalized[selected_indices],
                reference_points,
            )
            for association in selected_associations:
                niche_counts[association] += 1

        candidates = list(front)
        picked: list[int] = []
        while len(picked) < needed and candidates:
            minimum_count = min(
                niche_counts[associations[index]]
                for index in candidates
            )
            minimum_niches = {
                int(associations[index])
                for index in candidates
                if niche_counts[associations[index]] == minimum_count
            }
            niche_candidates = [
                index
                for index in candidates
                if int(associations[index]) in minimum_niches
            ]
            chosen = min(
                niche_candidates,
                key=lambda index: (
                    distances[index],
                    int(associations[index]),
                    index,
                ),
            )
            picked.append(chosen)
            candidates.remove(chosen)
            niche_counts[associations[chosen]] += 1
        selected_indices.extend(picked)
        break

    selected_objectives = objectives[selected_indices].copy()
    selected_fronts = _fast_non_dominated_sort(selected_objectives)
    return (
        [priorities[index].copy() for index in selected_indices],
        selected_objectives,
        selected_fronts,
    )


def _blend_crossover(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > crossover_probability:
        return first.copy(), second.copy()
    blend = rng.uniform(-0.18, 1.18, size=len(first))
    child_one = blend * first + (1.0 - blend) * second
    child_two = blend * second + (1.0 - blend) * first
    return child_one, child_two


def _mutate_priorities(
    priorities: np.ndarray,
    components: Sequence[Component],
    rng: np.random.Generator,
    mutation_probability: float,
) -> np.ndarray:
    mutated = priorities.copy()
    mask = rng.random(len(mutated)) <= mutation_probability
    mutated[mask] += rng.normal(0.0, 1.65, size=int(np.sum(mask)))
    layer_order = np.asarray([TYPE_SPECS[component.type]["layer"] for component in components])
    layer_shift = int(rng.integers(0, max(2, int(np.max(layer_order)) + 1)))
    if layer_shift:
        mutated += rng.normal(0.0, 0.38) * (layer_order == layer_shift)
    return mutated


def _normalise_objectives(objectives: np.ndarray) -> np.ndarray:
    minimum = objectives.min(axis=0)
    maximum = objectives.max(axis=0)
    span = np.where(maximum > minimum, maximum - minimum, 1.0)
    return (objectives - minimum) / span


def _choose_plan(
    objectives: np.ndarray,
    weights: Sequence[float],
    excluded_indices: set[int] | None = None,
    primary_objective: int | None = None,
) -> int:
    normalised = _normalise_objectives(objectives)
    score = normalised @ np.asarray(weights, dtype=float)
    excluded_indices = excluded_indices or set()
    if primary_objective is None:
        ordered_indices = np.argsort(score)
    else:
        ordered_indices = np.asarray(
            sorted(
                range(len(objectives)),
                key=lambda index: (
                    normalised[index, primary_objective],
                    score[index],
                ),
            )
        )
    for index in ordered_indices:
        if int(index) not in excluded_indices:
            return int(index)
    return int(np.argmin(score))


def optimize_hoisting_sequence(
    components: Sequence[Component],
    cranes: Sequence[dict],
    population_size: int = 40,
    generations: int = 30,
    crossover_probability: float = 0.82,
    mutation_probability: float = 0.10,
    seed: int = 20250925,
    decision_weights: Sequence[float] = (0.40, 0.30, 0.30),
    algorithm: str = "nsga2",
    objective_mode: str = "basic",
    reference_divisions: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    if population_size < 8:
        raise ValueError("population_size must be at least 8")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if algorithm not in {"nsga2", "nsga3"}:
        raise ValueError("algorithm must be 'nsga2' or 'nsga3'")
    if objective_mode not in {"basic", "expert"}:
        raise ValueError("objective_mode must be 'basic' or 'expert'")

    objective_count = 3 if objective_mode == "basic" else 5
    reference_points = (
        _generate_reference_points(objective_count, reference_divisions)
        if algorithm == "nsga3"
        else np.empty((0, objective_count), dtype=float)
    )

    rng = np.random.default_rng(seed)
    successors = component_successors(components)
    priorities = _initial_population(population_size, components, successors, rng)
    evaluation_cache: dict[tuple[int, ...], np.ndarray] = {}
    history = {
        "generation": [],
        "best_time": [],
        "best_cost": [],
        "best_risk": [],
        "front_size": [],
    }

    def evaluate_priority(priority: np.ndarray) -> np.ndarray:
        sequence = tuple(decode_priority_vector(priority, components, successors))
        if sequence not in evaluation_cache:
            metrics = schedule_sequence(sequence, components, cranes)
            objective = (
                metrics["expert_objective"]
                if objective_mode == "expert"
                else [
                    metrics["makespan_min"],
                    metrics["cost_yuan"],
                    metrics["risk_score"],
                ]
            )
            evaluation_cache[sequence] = np.asarray(objective, dtype=float)
        return evaluation_cache[sequence]

    objectives = np.asarray([evaluate_priority(priority) for priority in priorities])
    for generation in range(1, generations + 1):
        fronts = _fast_non_dominated_sort(objectives)
        rank = np.zeros(population_size, dtype=int)
        distance = np.zeros(population_size, dtype=float)
        for front_index, front in enumerate(fronts):
            front_distance = _crowding_distance(objectives, front)
            for index in front:
                rank[index] = front_index
                distance[index] = front_distance[index]

        offspring: list[np.ndarray] = []
        while len(offspring) < population_size:
            first_candidates = rng.integers(0, population_size, size=2)
            second_candidates = rng.integers(0, population_size, size=2)
            first = min(
                first_candidates,
                key=lambda index: (rank[index], -distance[index]),
            )
            second = min(
                second_candidates,
                key=lambda index: (rank[index], -distance[index]),
            )
            child_one, child_two = _blend_crossover(
                priorities[first],
                priorities[second],
                rng,
                crossover_probability,
            )
            offspring.append(
                _mutate_priorities(
                    child_one,
                    components,
                    rng,
                    mutation_probability,
                )
            )
            if len(offspring) < population_size:
                offspring.append(
                    _mutate_priorities(
                        child_two,
                        components,
                        rng,
                        mutation_probability,
                    )
                )

        offspring_objectives = np.asarray(
            [evaluate_priority(priority) for priority in offspring]
        )
        combined_priorities = priorities + offspring
        combined_objectives = np.vstack([objectives, offspring_objectives])
        if algorithm == "nsga3":
            priorities, objectives, selected_fronts = _select_nsga3(
                combined_priorities,
                combined_objectives,
                population_size,
                reference_points,
            )
        else:
            priorities, objectives, selected_fronts = _select_next_generation(
                combined_priorities,
                combined_objectives,
                population_size,
            )

        history["generation"].append(generation)
        history["best_time"].append(float(objectives[:, 0].min()))
        history["best_cost"].append(float(objectives[:, 1].min()))
        history["best_risk"].append(float(objectives[:, 2].min()))
        if objective_mode == "expert":
            history.setdefault("best_spatial_risk", []).append(
                float(objectives[:, 2].min())
            )
            history.setdefault("best_environment_risk", []).append(
                float(objectives[:, 3].min())
            )
            history.setdefault("best_equipment_balance", []).append(
                float(objectives[:, 4].min())
            )
        history["front_size"].append(len(selected_fronts[0]))
        if progress_callback is not None:
            progress_callback(generation, generations, objectives)

    final_fronts = _fast_non_dominated_sort(objectives)
    front_indices = final_fronts[0]
    front_priorities = [priorities[index].copy() for index in front_indices]
    front_objectives = objectives[front_indices].copy()

    unique_rows: dict[tuple[float, ...], int] = {}
    for row_index, row in enumerate(front_objectives):
        key = tuple(np.round(row, 5))
        unique_rows.setdefault(key, row_index)
    keep_indices = list(unique_rows.values())
    if len(keep_indices) < 3:
        keep_indices = list(range(min(3, len(front_objectives))))
    front_priorities = [front_priorities[index] for index in keep_indices]
    front_objectives = front_objectives[keep_indices]

    if objective_mode == "expert":
        short, medium = float(decision_weights[0]), float(decision_weights[1])
        safety = float(decision_weights[2])
        plan_definitions = {
            "工期优先": ((0.72, 0.08, 0.08, 0.06, 0.06), 0),
            "成本优先": ((0.08, 0.72, 0.08, 0.06, 0.06), 1),
            "空间安全": ((0.10, 0.10, 0.60, 0.10, 0.10), 2),
            "环境安全": ((0.10, 0.10, 0.10, 0.60, 0.10), 3),
            "设备均衡": ((0.10, 0.10, 0.10, 0.10, 0.60), 4),
            "综合推荐": (
                (
                    short,
                    medium,
                    safety * 0.50,
                    safety * 0.35,
                    safety * 0.15,
                ),
                None,
            ),
        }
    else:
        plan_definitions = {
            "A · 进度优先": ((0.70, 0.20, 0.10), 0),
            "B · 成本优先": ((0.15, 0.75, 0.10), 1),
            "C · 安全优先": ((0.10, 0.15, 0.75), 2),
            "综合推荐": (
                tuple(float(value) for value in decision_weights),
                None,
            ),
        }
    plans: dict[str, dict] = {}
    selected_plan_indices: set[int] = set()
    for plan_name, (weights, primary_objective) in plan_definitions.items():
        selected = _choose_plan(
            front_objectives,
            weights,
            excluded_indices=(
                selected_plan_indices
                if plan_name != "综合推荐"
                else None
            ),
            primary_objective=primary_objective,
        )
        selected_plan_indices.add(selected)
        sequence = decode_priority_vector(
            front_priorities[selected],
            components,
            successors,
        )
        metrics = schedule_sequence(sequence, components, cranes)
        metrics["weight_vector"] = tuple(weights)
        metrics["objective"] = front_objectives[selected].tolist()
        metrics["plan_name"] = plan_name
        plans[plan_name] = metrics

    return OptimizationResult(
        components=list(components),
        pareto_priorities=front_priorities,
        pareto_objectives=front_objectives,
        history=history,
        plans=plans,
        cranes=list(cranes),
        metadata={
            "population_size": population_size,
            "generations": generations,
            "evaluations": len(evaluation_cache),
            "pareto_size": len(front_objectives),
            "seed": seed,
            "algorithm": algorithm.upper(),
            "objective_mode": objective_mode,
            "objective_names": (
                [
                    "工期",
                    "成本",
                    "空间碰撞",
                    "作业环境",
                    "设备均衡",
                ]
                if objective_mode == "expert"
                else ["工期", "成本", "综合安全"]
            ),
            "reference_points": len(reference_points),
            "reference_divisions": reference_divisions,
        },
    )
