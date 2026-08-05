from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from glioma_shift_atlas.contracts import InferenceRecord, PatientBatch, PatientOutput


@dataclass(frozen=True)
class LatencyRecord:
    patient_key: str
    seconds: float
    peak_memory_bytes: int
    modality_count: int


@dataclass(frozen=True)
class DeploymentSummary:
    patients: int
    median_seconds: float
    mean_seconds: float
    standard_deviation_seconds: float
    patients_per_minute: float
    maximum_memory_bytes: int
    low_confidence_patients: int


def probability(logit: Tensor) -> Tensor:
    return torch.sigmoid(logit)


def output_records(
    batch: PatientBatch,
    output: PatientOutput,
    confidence_threshold: float,
) -> list[InferenceRecord]:
    shift = output.shift.detach().cpu().tolist()
    subtype = output.subtype.detach().cpu().tolist()
    confidence = output.confidence.detach().cpu().tolist()
    risk = output.risk.detach().cpu().tolist()
    therapy = probability(output.therapy_logit).detach().cpu().tolist()
    records: list[InferenceRecord] = []
    for index, patient_key in enumerate(batch.patient_keys):
        records.append(
            InferenceRecord(
                patient_key=patient_key,
                subtype=int(subtype[index]) + 1,
                confidence=float(confidence[index]),
                compositional_shift=tuple(float(value) for value in shift[index]),
                survival_risk=float(risk[index]),
                therapy_probability=float(therapy[index]),
                low_confidence=float(confidence[index]) < confidence_threshold,
            )
        )
    return records


def route_label(record: InferenceRecord) -> str:
    return "multidisciplinary_review" if record.low_confidence else "standard_tumor_board"


def clinical_payload(record: InferenceRecord) -> dict[str, object]:
    return {
        "patient_key": record.patient_key,
        "microenvironment_subtype": f"S{record.subtype}",
        "assignment_confidence": record.confidence,
        "compositional_shift": list(record.compositional_shift),
        "survival_risk": record.survival_risk,
        "therapy_response_probability": record.therapy_probability,
        "review_route": route_label(record),
    }


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_inference_records(records: Sequence[InferenceRecord], destination: Path) -> None:
    payload = [clinical_payload(record) for record in records]
    atomic_json(payload, destination)


@torch.no_grad()
def timed_inference(
    model: nn.Module,
    batch: PatientBatch,
    device: torch.device,
) -> tuple[PatientOutput, list[LatencyRecord]]:
    model.eval()
    moved = batch.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(moved)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    per_patient = elapsed / max(1, batch.size)
    records = [
        LatencyRecord(
            patient_key=key,
            seconds=per_patient,
            peak_memory_bytes=memory,
            modality_count=int(batch.modality_mask[index].sum().item()),
        )
        for index, key in enumerate(batch.patient_keys)
    ]
    return output, records


def deployment_summary(
    latency: Sequence[LatencyRecord],
    inference: Sequence[InferenceRecord],
) -> DeploymentSummary:
    seconds = np.asarray([item.seconds for item in latency], dtype=np.float64)
    median = float(np.median(seconds)) if seconds.shape[0] else 0.0
    mean = float(np.mean(seconds)) if seconds.shape[0] else 0.0
    deviation = float(np.std(seconds, ddof=1)) if seconds.shape[0] > 1 else 0.0
    throughput = 60.0 / median if median > 0 else 0.0
    maximum_memory = max((item.peak_memory_bytes for item in latency), default=0)
    low_confidence = sum(item.low_confidence for item in inference)
    return DeploymentSummary(
        patients=len(latency),
        median_seconds=median,
        mean_seconds=mean,
        standard_deviation_seconds=deviation,
        patients_per_minute=throughput,
        maximum_memory_bytes=maximum_memory,
        low_confidence_patients=low_confidence,
    )


def export_torchscript(model: nn.Module, example: PatientBatch, path: Path) -> None:
    model.eval()
    traced = torch.jit.trace(model, (example,), strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path))


def quantize_dynamic(model: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def confidence_histogram(records: Sequence[InferenceRecord], bins: int = 10) -> dict[str, list[float]]:
    values = np.asarray([record.confidence for record in records], dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {
        "edges": edges.tolist(),
        "counts": counts.astype(np.float64).tolist(),
    }


def subtype_distribution(records: Sequence[InferenceRecord]) -> dict[str, float]:
    total = max(1, len(records))
    return {
        f"S{subtype}": sum(record.subtype == subtype for record in records) / total
        for subtype in range(1, 7)
    }


def drift_score(
    current: Sequence[InferenceRecord],
    reference: Sequence[InferenceRecord],
) -> float:
    current_distribution = subtype_distribution(current)
    reference_distribution = subtype_distribution(reference)
    current_values = np.asarray(list(current_distribution.values()), dtype=np.float64)
    reference_values = np.asarray(list(reference_distribution.values()), dtype=np.float64)
    midpoint = 0.5 * (current_values + reference_values)
    left = np.sum(current_values * np.log(np.clip(current_values / midpoint, 1e-12, None)))
    right = np.sum(reference_values * np.log(np.clip(reference_values / midpoint, 1e-12, None)))
    return float(0.5 * (left + right))


def audit_payload(
    records: Sequence[InferenceRecord],
    latency: Sequence[LatencyRecord],
) -> dict[str, object]:
    summary = deployment_summary(latency, records)
    return {
        "deployment": asdict(summary),
        "confidence": confidence_histogram(records),
        "subtypes": subtype_distribution(records),
    }
