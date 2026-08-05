from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RunResult:
    experiment: str
    cohort: str
    seed: int
    fold: int
    metric: str
    value: float
    patients: int
    duration_seconds: float
    status: str


@dataclass(frozen=True)
class AggregateResult:
    experiment: str
    cohort: str
    metric: str
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    runs: int
    patients: int
    duration_seconds: float


def load_result(path: Path) -> RunResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunResult(
        experiment=str(payload["experiment"]),
        cohort=str(payload["cohort"]),
        seed=int(payload["seed"]),
        fold=int(payload["fold"]),
        metric=str(payload["metric"]),
        value=float(payload["value"]),
        patients=int(payload["patients"]),
        duration_seconds=float(payload["duration_seconds"]),
        status=str(payload["status"]),
    )


def discover_results(root: Path) -> list[RunResult]:
    output: list[RunResult] = []
    for path in sorted(root.rglob("result.json")):
        result = load_result(path)
        if result.status == "complete":
            output.append(result)
    return output


def aggregate_results(results: Sequence[RunResult]) -> list[AggregateResult]:
    grouped: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[(result.experiment, result.cohort, result.metric)].append(result)
    output: list[AggregateResult] = []
    for (experiment, cohort, metric), group in sorted(grouped.items()):
        values = np.asarray([item.value for item in group], dtype=np.float64)
        output.append(
            AggregateResult(
                experiment=experiment,
                cohort=cohort,
                metric=metric,
                mean=float(np.mean(values)),
                standard_deviation=float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0,
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                runs=len(group),
                patients=sum(item.patients for item in group),
                duration_seconds=sum(item.duration_seconds for item in group),
            )
        )
    return output


def write_aggregates(results: Sequence[AggregateResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "cohort",
                "metric",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
                "runs",
                "patients",
                "duration_seconds",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.experiment,
                    result.cohort,
                    result.metric,
                    result.mean,
                    result.standard_deviation,
                    result.minimum,
                    result.maximum,
                    result.runs,
                    result.patients,
                    result.duration_seconds,
                ]
            )


def compare_to_reference(
    results: Sequence[AggregateResult],
    reference: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for result in results:
        key = (result.cohort, result.metric)
        expected = reference.get(key)
        if expected is not None:
            output[f"{result.cohort}:{result.metric}"] = result.mean - expected
    return output


def seed_stability(results: Sequence[RunResult]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        grouped[result.experiment].append(result.value)
    return {
        experiment: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for experiment, values in grouped.items()
    }


def fold_stability(results: Sequence[RunResult]) -> dict[str, float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for result in results:
        grouped[(result.experiment, result.fold)].append(result.value)
    output: dict[str, float] = {}
    experiments = sorted({result.experiment for result in results})
    for experiment in experiments:
        fold_means = [
            np.mean(values)
            for (name, _), values in grouped.items()
            if name == experiment
        ]
        output[experiment] = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
    return output


def total_gpu_hours(results: Iterable[RunResult], world_size: int) -> float:
    seconds = sum(result.duration_seconds for result in results)
    return seconds * world_size / 3600.0
