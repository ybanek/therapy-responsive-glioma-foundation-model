from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from glioma_shift_atlas.contracts import AtlasConfig, InferenceRecord, PatientBatch, PatientOutput
from glioma_shift_atlas.engine import all_gather_variable, autocast_context
from glioma_shift_atlas.statistics import (
    ClassificationReport,
    SurvivalReport,
    as_float,
    as_int,
    bootstrap_auc,
    bootstrap_concordance,
    bootstrap_kappa,
    classification_report,
    cohen_kappa,
    cochran_q,
    integrated_discrimination_improvement,
    macro_f1,
    net_reclassification_improvement,
    random_effects_meta_analysis,
    survival_report,
)


@dataclass
class PredictionTable:
    patient_keys: list[str]
    cohorts: list[str]
    survival_time: list[float]
    survival_event: list[int]
    therapy_target: list[int]
    therapy_observed: list[bool]
    subtype_target: list[int]
    subtype_prediction: list[int]
    confidence: list[float]
    risk: list[float]
    therapy_probability: list[float]
    shift_prediction: list[list[float]]
    prototype_probability: list[list[float]]

    @classmethod
    def empty(cls) -> PredictionTable:
        return cls([], [], [], [], [], [], [], [], [], [], [], [], [], [])

    def extend(self, batch: PatientBatch, output: PatientOutput) -> None:
        self.patient_keys.extend(batch.patient_keys)
        self.cohorts.extend(batch.cohorts)
        self.survival_time.extend(batch.survival_time.detach().cpu().tolist())
        self.survival_event.extend(batch.survival_event.detach().cpu().tolist())
        self.therapy_target.extend(batch.therapy_response.detach().cpu().tolist())
        self.therapy_observed.extend(batch.therapy_observed.detach().cpu().tolist())
        self.subtype_target.extend(batch.subtype_scores.argmax(dim=-1).detach().cpu().tolist())
        self.subtype_prediction.extend(output.subtype.detach().cpu().tolist())
        self.confidence.extend(output.confidence.detach().cpu().tolist())
        self.risk.extend(output.risk.detach().cpu().tolist())
        probability = torch.sigmoid(output.therapy_logit)
        self.therapy_probability.extend(probability.detach().cpu().tolist())
        self.shift_prediction.extend(output.shift.detach().cpu().tolist())
        self.prototype_probability.extend(output.prototype_probabilities.detach().cpu().tolist())

    def subset(self, indices: Sequence[int]) -> PredictionTable:
        names = tuple(self.__dataclass_fields__)
        values: dict[str, object] = {}
        for name in names:
            column = getattr(self, name)
            values[name] = [column[index] for index in indices]
        return PredictionTable(**values)

    def cohort(self, cohort: str) -> PredictionTable:
        indices = [index for index, value in enumerate(self.cohorts) if value == cohort]
        return self.subset(indices)

    def size(self) -> int:
        return len(self.patient_keys)


@dataclass(frozen=True)
class CohortReport:
    cohort: str
    patients: int
    survival: SurvivalReport
    therapy: ClassificationReport | None
    c_index_lower: float
    c_index_upper: float
    auc_lower: float | None
    auc_upper: float | None
    subtype_kappa: float
    subtype_kappa_lower: float
    subtype_kappa_upper: float
    subtype_macro_f1: float
    low_confidence_fraction: float


def evaluate_cohort(table: PredictionTable, config: AtlasConfig, seed: int) -> CohortReport:
    time = as_float(table.survival_time)
    event = as_int(table.survival_event)
    risk = as_float(table.risk)
    subtype = as_int(table.subtype_prediction)
    subtype_target = as_int(table.subtype_target)
    high_risk = (risk >= np.median(risk)).astype(np.int64)
    survival = survival_report(time, event, risk, high_risk)
    c_index = bootstrap_concordance(
        time,
        risk,
        event,
        samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence,
        seed=seed,
    )
    observed = np.asarray(table.therapy_observed, dtype=bool)
    therapy = None
    auc_lower = None
    auc_upper = None
    if np.sum(observed) > 1:
        target = as_int(np.asarray(table.therapy_target)[observed])
        probability = as_float(np.asarray(table.therapy_probability)[observed])
        if np.unique(target).shape[0] > 1:
            therapy = classification_report(target, probability)
            auc = bootstrap_auc(
                target,
                probability,
                samples=config.evaluation.bootstrap_samples,
                confidence=config.evaluation.confidence,
                seed=seed,
            )
            auc_lower = auc.lower
            auc_upper = auc.upper
    kappa = bootstrap_kappa(
        subtype_target,
        subtype,
        samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence,
        seed=seed,
    )
    name = table.cohorts[0] if table.cohorts else "unknown"
    return CohortReport(
        cohort=name,
        patients=table.size(),
        survival=survival,
        therapy=therapy,
        c_index_lower=c_index.lower,
        c_index_upper=c_index.upper,
        auc_lower=auc_lower,
        auc_upper=auc_upper,
        subtype_kappa=cohen_kappa(subtype_target, subtype),
        subtype_kappa_lower=kappa.lower,
        subtype_kappa_upper=kappa.upper,
        subtype_macro_f1=macro_f1(subtype_target, subtype),
        low_confidence_fraction=float(
            np.mean(as_float(table.confidence) < config.model.confidence_threshold)
        ),
    )


@dataclass(frozen=True)
class HeterogeneityReport:
    cochran_q: float
    cochran_p: float
    i_squared: float
    tau_squared: float
    pooled: float
    pooled_lower: float
    pooled_upper: float


def heterogeneity(reports: Sequence[CohortReport]) -> HeterogeneityReport:
    estimates = as_float([report.survival.concordance for report in reports])
    standard_errors = as_float(
        [
            max(1e-6, (report.c_index_upper - report.c_index_lower) / (2.0 * 1.96))
            for report in reports
        ]
    )
    variances = standard_errors**2
    q, p_value, i_squared, tau_squared = cochran_q(estimates, variances)
    pooled, lower, upper = random_effects_meta_analysis(estimates, variances)
    return HeterogeneityReport(q, p_value, i_squared, tau_squared, pooled, lower, upper)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader[PatientBatch],
    device: torch.device,
    precision: str,
) -> PredictionTable:
    model.eval()
    table = PredictionTable.empty()
    for batch in loader:
        moved = batch.to(device)
        with autocast_context(device, precision):
            output = model(moved)
        table.extend(moved, output)
    return table


def inference_records(table: PredictionTable, threshold: float) -> list[InferenceRecord]:
    records: list[InferenceRecord] = []
    for index, key in enumerate(table.patient_keys):
        confidence = table.confidence[index]
        records.append(
            InferenceRecord(
                patient_key=key,
                subtype=table.subtype_prediction[index] + 1,
                confidence=confidence,
                compositional_shift=tuple(table.shift_prediction[index]),
                survival_risk=table.risk[index],
                therapy_probability=table.therapy_probability[index],
                low_confidence=confidence < threshold,
            )
        )
    return records


def write_prediction_csv(table: PredictionTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_key",
                "cohort",
                "survival_time",
                "survival_event",
                "therapy_target",
                "therapy_observed",
                "subtype_target",
                "subtype_prediction",
                "confidence",
                "risk",
                "therapy_probability",
                "shift_prediction",
                "prototype_probability",
            ]
        )
        for index in range(table.size()):
            writer.writerow(
                [
                    table.patient_keys[index],
                    table.cohorts[index],
                    table.survival_time[index],
                    table.survival_event[index],
                    table.therapy_target[index],
                    table.therapy_observed[index],
                    table.subtype_target[index],
                    table.subtype_prediction[index],
                    table.confidence[index],
                    table.risk[index],
                    table.therapy_probability[index],
                    json.dumps(table.shift_prediction[index]),
                    json.dumps(table.prototype_probability[index]),
                ]
            )


def write_report(
    reports: Sequence[CohortReport],
    heterogeneity_report: HeterogeneityReport,
    path: Path,
) -> None:
    payload = {
        "cohorts": [asdict(report) for report in reports],
        "heterogeneity": asdict(heterogeneity_report),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def threshold_impact(
    target: Sequence[int],
    probability: Sequence[float],
    reference: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    truth = as_int(target)
    proposed = (as_float(probability) >= threshold).astype(np.int64)
    baseline = (as_float(reference) >= threshold).astype(np.int64)
    proposed_correct = np.mean(proposed == truth)
    baseline_correct = np.mean(baseline == truth)
    additional = (proposed_correct - baseline_correct) * 1000.0
    false_positive = np.mean((proposed == 1) & (truth == 0))
    nri = net_reclassification_improvement(
        truth,
        as_float(probability),
        as_float(reference),
        (threshold,),
    )
    idi = integrated_discrimination_improvement(
        truth,
        as_float(probability),
        as_float(reference),
    )
    return {
        "additional_correct_per_1000": float(additional),
        "false_positive_burden": float(false_positive),
        "net_reclassification_improvement": nri,
        "integrated_discrimination_improvement": idi,
    }
