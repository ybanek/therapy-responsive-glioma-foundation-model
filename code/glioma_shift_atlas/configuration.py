from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from glioma_shift_atlas.contracts import (
    AtlasConfig,
    DataConfig,
    DistributedConfig,
    EvaluationConfig,
    FinetuneConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainConfig,
    RuntimeConfig,
)

T = TypeVar("T")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _value(mapping: dict[str, Any], name: str, expected: type[T]) -> T:
    value = mapping.get(name)
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")
    return value


def _number(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _integer(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _strings(mapping: dict[str, Any], name: str) -> tuple[str, ...]:
    value = mapping.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain strings")
    return tuple(value)


def _integers(mapping: dict[str, Any], name: str) -> tuple[int, ...]:
    value = mapping.get(name)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain integers")
    return tuple(value)


def load_config(path: Path) -> AtlasConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "configuration")
    data = _mapping(root.get("data"), "data")
    model = _mapping(root.get("model"), "model")
    pretrain = _mapping(root.get("pretrain"), "pretrain")
    finetune = _mapping(root.get("finetune"), "finetune")
    optimizer = _mapping(root.get("optimizer"), "optimizer")
    distributed = _mapping(root.get("distributed"), "distributed")
    loss = _mapping(root.get("loss"), "loss")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    runtime = _mapping(root.get("runtime"), "runtime")
    config = AtlasConfig(
        seed=_integer(root, "seed"),
        seeds=_integers(root, "seeds"),
        data=DataConfig(
            manifest=Path(_value(data, "manifest", str)),
            cohorts=_strings(data, "cohorts"),
            folds=_integer(data, "folds"),
            train_fraction=_number(data, "train_fraction"),
            validation_fraction=_number(data, "validation_fraction"),
            test_fraction=_number(data, "test_fraction"),
            visium_radius_microns=_number(data, "visium_radius_microns"),
            xenium_radius_microns=_number(data, "xenium_radius_microns"),
            patch_magnification=_integer(data, "patch_magnification"),
            patch_size=_integer(data, "patch_size"),
        ),
        model=ModelConfig(
            patient_dim=_integer(model, "patient_dim"),
            clinical_dim=_integer(model, "clinical_dim"),
            clinical_hidden=_integer(model, "clinical_hidden"),
            clinical_output=_integer(model, "clinical_output"),
            histology_output=_integer(model, "histology_output"),
            spatial_output=_integer(model, "spatial_output"),
            transcript_output=_integer(model, "transcript_output"),
            set_layers=_integer(model, "set_layers"),
            set_heads=_integer(model, "set_heads"),
            prototypes=_integer(model, "prototypes"),
            experts=_integer(model, "experts"),
            idh_experts=_integer(model, "idh_experts"),
            confidence_threshold=_number(model, "confidence_threshold"),
            ib_beta=_number(model, "ib_beta"),
        ),
        pretrain=PretrainConfig(
            epochs=_integer(pretrain, "epochs"),
            warmup_epochs=_integer(pretrain, "warmup_epochs"),
            dropout_ramp_end=_integer(pretrain, "dropout_ramp_end"),
            dropout_final=_number(pretrain, "dropout_final"),
            paired_weight=_number(pretrain, "paired_weight"),
            pseudo_shift_weight=_number(pretrain, "pseudo_shift_weight"),
        ),
        finetune=FinetuneConfig(epochs=_integer(finetune, "epochs")),
        optimizer=OptimizerConfig(
            name=_value(optimizer, "name", str),
            learning_rate=_number(optimizer, "learning_rate"),
            weight_decay=_number(optimizer, "weight_decay"),
            batch_size=_integer(optimizer, "batch_size"),
            gradient_accumulation=_integer(optimizer, "gradient_accumulation"),
            gradient_clip_norm=_number(optimizer, "gradient_clip_norm"),
            scheduler=_value(optimizer, "scheduler", str),
            precision=_value(optimizer, "precision", str),
        ),
        distributed=DistributedConfig(
            world_size=_integer(distributed, "world_size"),
            backend=_value(distributed, "backend", str),
        ),
        loss=LossConfig(
            shift=_number(loss, "shift"),
            clinical=_number(loss, "clinical"),
            alignment=_number(loss, "alignment"),
            modality=_number(loss, "modality"),
        ),
        evaluation=EvaluationConfig(
            bootstrap_samples=_integer(evaluation, "bootstrap_samples"),
            permutation_samples=_integer(evaluation, "permutation_samples"),
            confidence=_number(evaluation, "confidence"),
            horizons_months=_integers(evaluation, "horizons_months"),
        ),
        runtime=RuntimeConfig(
            output=Path(_value(runtime, "output", str)),
            workers=_integer(runtime, "workers"),
            pin_memory=_value(runtime, "pin_memory", bool),
            persistent_workers=_value(runtime, "persistent_workers", bool),
            atomic_checkpoints=_value(runtime, "atomic_checkpoints", bool),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: AtlasConfig) -> None:
    fractions = (
        config.data.train_fraction
        + config.data.validation_fraction
        + config.data.test_fraction
    )
    if abs(fractions - 1.0) > 1e-8:
        raise ValueError("data fractions must sum to one")
    if config.model.patient_dim % config.model.set_heads != 0:
        raise ValueError("patient dimension must be divisible by attention heads")
    if config.model.prototypes < 2:
        raise ValueError("at least two prototypes are required")
    if config.pretrain.warmup_epochs >= config.pretrain.dropout_ramp_end:
        raise ValueError("dropout ramp must end after warmup")
    if not 0.0 <= config.pretrain.dropout_final < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    if config.optimizer.batch_size < 1:
        raise ValueError("batch size must be positive")
    if config.distributed.world_size < 1:
        raise ValueError("world size must be positive")
    if not 0.0 < config.evaluation.confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
