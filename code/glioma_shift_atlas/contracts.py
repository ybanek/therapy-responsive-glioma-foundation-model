from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from torch import Tensor

ModalityName = Literal["histology", "spatial", "single_cell", "bulk", "clinical"]
StageName = Literal["pretrain", "finetune", "evaluate", "infer"]


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    cohorts: tuple[str, ...]
    folds: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    visium_radius_microns: float
    xenium_radius_microns: float
    patch_magnification: int
    patch_size: int


@dataclass(frozen=True)
class ModelConfig:
    patient_dim: int
    clinical_dim: int
    clinical_hidden: int
    clinical_output: int
    histology_output: int
    spatial_output: int
    transcript_output: int
    set_layers: int
    set_heads: int
    prototypes: int
    experts: int
    idh_experts: int
    confidence_threshold: float
    ib_beta: float


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int
    warmup_epochs: int
    dropout_ramp_end: int
    dropout_final: float
    paired_weight: float
    pseudo_shift_weight: float


@dataclass(frozen=True)
class FinetuneConfig:
    epochs: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation: int
    gradient_clip_norm: float
    scheduler: str
    precision: str


@dataclass(frozen=True)
class DistributedConfig:
    world_size: int
    backend: str


@dataclass(frozen=True)
class LossConfig:
    shift: float
    clinical: float
    alignment: float
    modality: float


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int
    permutation_samples: int
    confidence: float
    horizons_months: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    output: Path
    workers: int
    pin_memory: bool
    persistent_workers: bool
    atomic_checkpoints: bool


@dataclass(frozen=True)
class AtlasConfig:
    seed: int
    seeds: tuple[int, ...]
    data: DataConfig
    model: ModelConfig
    pretrain: PretrainConfig
    finetune: FinetuneConfig
    optimizer: OptimizerConfig
    distributed: DistributedConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class PatientRecord:
    patient_key: str
    cohort: str
    histology_path: Path | None
    spatial_path: Path | None
    single_cell_path: Path | None
    bulk_path: Path | None
    clinical_path: Path | None
    overall_survival_months: float
    event: int
    therapy_response: int | None
    idh_status: int | None
    mgmt_status: int | None
    subtype_scores: tuple[float, float, float, float]
    shift_path: Path | None


@dataclass
class PatientBatch:
    patient_keys: list[str]
    cohorts: list[str]
    histology: Tensor | None
    histology_mask: Tensor | None
    spatial_expression: Tensor | None
    spatial_coordinates: Tensor | None
    spatial_mask: Tensor | None
    single_cell_expression: Tensor | None
    single_cell_mask: Tensor | None
    bulk_expression: Tensor | None
    clinical: Tensor | None
    modality_mask: Tensor
    survival_time: Tensor
    survival_event: Tensor
    therapy_response: Tensor
    therapy_observed: Tensor
    idh_status: Tensor
    idh_observed: Tensor
    subtype_scores: Tensor
    compositional_shift: Tensor
    shift_observed: Tensor

    def to(self, device: torch.device, non_blocking: bool = True) -> PatientBatch:
        values: dict[str, object] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                values[key] = value.to(device=device, non_blocking=non_blocking)
            else:
                values[key] = value
        return PatientBatch(**values)

    @property
    def size(self) -> int:
        return len(self.patient_keys)


class EncodedModalities(NamedTuple):
    tokens: Tensor
    availability: Tensor
    modality_names: tuple[str, ...]


@dataclass
class PatientOutput:
    embedding: Tensor
    fused_embedding: Tensor
    idh_embedding: Tensor
    prototype_logits: Tensor
    prototype_probabilities: Tensor
    subtype: Tensor
    confidence: Tensor
    shift: Tensor
    risk: Tensor
    therapy_logit: Tensor
    idh_logit: Tensor
    variational_mean: Tensor
    variational_log_variance: Tensor
    routing_weights: Tensor


@dataclass
class LossBundle:
    total: Tensor
    subtype: Tensor
    shift: Tensor
    cox: Tensor
    therapy: Tensor
    alignment: Tensor
    modality: Tensor
    kl: Tensor
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitAssignment:
    patient_key: str
    cohort: str
    fold: int
    partition: Literal["train", "validation", "test"]


@dataclass(frozen=True)
class EpochState:
    stage: StageName
    epoch: int
    global_step: int
    best_metric: float
    seed: int


@dataclass(frozen=True)
class MetricEstimate:
    name: str
    value: float
    lower: float
    upper: float
    standard_deviation: float
    sample_count: int


@dataclass(frozen=True)
class SubgroupEstimate:
    metric: str
    subgroup: str
    level: str
    value: float
    lower: float
    upper: float
    sample_count: int


@dataclass(frozen=True)
class InferenceRecord:
    patient_key: str
    subtype: int
    confidence: float
    compositional_shift: tuple[float, ...]
    survival_risk: float
    therapy_probability: float
    low_confidence: bool
