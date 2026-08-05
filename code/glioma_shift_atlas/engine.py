from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from glioma_shift_atlas.contracts import AtlasConfig, EpochState, LossBundle, PatientBatch
from glioma_shift_atlas.model import GliomaShiftAtlas
from glioma_shift_atlas.objectives import (
    FinetuningObjective,
    ObjectiveWeights,
    PretrainingObjective,
    modality_dropout_probability,
    pseudo_shift_targets,
    sample_modality_mask,
)

LOGGER = logging.getLogger("glioma_shift_atlas")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_determinism(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)


def distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if distributed_available() else 0


def world_size() -> int:
    return dist.get_world_size() if distributed_available() else 1


def is_primary() -> bool:
    return rank() == 0


def initialize_distributed(backend: str) -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not distributed_available():
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def finalize_distributed() -> None:
    if distributed_available():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: Tensor) -> Tensor:
    if not distributed_available():
        return value
    output = value.detach().clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    output /= world_size()
    return output


def all_gather_variable(value: Tensor) -> Tensor:
    if not distributed_available():
        return value
    local_size = torch.tensor([value.shape[0]], device=value.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size())]
    dist.all_gather(sizes, local_size)
    maximum = max(int(item.item()) for item in sizes)
    padding = maximum - value.shape[0]
    if padding > 0:
        shape = (padding, *value.shape[1:])
        value = torch.cat((value, value.new_zeros(shape)), dim=0)
    gathered = [torch.zeros_like(value) for _ in range(world_size())]
    dist.all_gather(gathered, value)
    trimmed = [item[: int(size.item())] for item, size in zip(gathered, sizes)]
    return torch.cat(trimmed, dim=0)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_distributed(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if distributed_available():
        device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    return model


def autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model: nn.Module, config: AtlasConfig) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(
        parameters,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(optimizer: Optimizer, epochs: int) -> LRScheduler:
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)


@dataclass
class RunningAverage:
    total: float = 0.0
    weight: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += value * weight
        self.weight += weight

    @property
    def mean(self) -> float:
        return self.total / self.weight if self.weight else 0.0


class MetricLedger:
    def __init__(self) -> None:
        self.values: dict[str, RunningAverage] = defaultdict(RunningAverage)

    def update(self, values: Mapping[str, float], weight: int = 1) -> None:
        for name, value in values.items():
            self.values[name].update(value, weight)

    def means(self) -> dict[str, float]:
        return {name: average.mean for name, average in self.values.items()}

    def distributed_means(self, device: torch.device) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, average in self.values.items():
            pair = torch.tensor([average.total, average.weight], device=device, dtype=torch.float64)
            if distributed_available():
                dist.all_reduce(pair, op=dist.ReduceOp.SUM)
            output[name] = float(pair[0] / pair[1].clamp_min(1.0))
        return output


@dataclass(frozen=True)
class CheckpointPayload:
    state: EpochState
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    random_state: object
    numpy_state: tuple[Any, ...]
    torch_state: Tensor
    cuda_states: list[Tensor]


def checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    state: EpochState,
) -> dict[str, Any]:
    module = unwrap_model(model)
    return {
        "state": asdict(state),
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    state: EpochState,
    path: Path,
) -> None:
    if is_primary():
        atomic_torch_save(checkpoint_payload(model, optimizer, scheduler, scaler, state), path)
    if distributed_available():
        dist.barrier()


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    path: Path,
    device: torch.device,
) -> EpochState:
    payload = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["random_state"])
    np.random.set_state(payload["numpy_state"])
    torch.set_rng_state(payload["torch_state"])
    if torch.cuda.is_available() and payload["cuda_states"]:
        torch.cuda.set_rng_state_all(payload["cuda_states"])
    state = payload["state"]
    set_seed(int(state["seed"]))
    return EpochState(
        stage=state["stage"],
        epoch=int(state["epoch"]),
        global_step=int(state["global_step"]),
        best_metric=float(state["best_metric"]),
        seed=int(state["seed"]),
    )


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


def finite_loss(bundle: LossBundle) -> None:
    if not torch.isfinite(bundle.total):
        raise FloatingPointError(f"non-finite loss: {bundle.values}")


class Trainer:
    def __init__(
        self,
        model: GliomaShiftAtlas,
        config: AtlasConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.model = wrap_distributed(model, device)
        self.optimizer = build_optimizer(self.model, config)
        self.scaler = GradScaler(enabled=device.type == "cuda" and config.optimizer.precision == "fp16")
        weights = ObjectiveWeights.from_config(config.loss, config.model.ib_beta)
        self.pretraining_objective = PretrainingObjective(weights).to(device)
        self.finetuning_objective = FinetuningObjective(weights).to(device)
        self.global_step = 0

    def _backward(self, loss: Tensor, accumulation: int) -> None:
        self.scaler.scale(loss / accumulation).backward()

    def _step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optimizer.gradient_clip_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(norm)

    def pretrain_epoch(self, loader: DataLoader[PatientBatch], epoch: int) -> dict[str, float]:
        self.model.train()
        ledger = MetricLedger()
        probability = modality_dropout_probability(
            epoch,
            self.config.pretrain.warmup_epochs,
            self.config.pretrain.dropout_ramp_end,
            self.config.pretrain.dropout_final,
        )
        accumulation = self.config.optimizer.gradient_accumulation
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            sampled = sample_modality_mask(batch.modality_mask, probability)
            target, weight = pseudo_shift_targets(
                batch.compositional_shift,
                batch.shift_observed,
                batch.subtype_scores,
                batch.therapy_response,
                torch.zeros_like(batch.idh_status),
                batch.idh_status,
            )
            with autocast_context(self.device, self.config.optimizer.precision):
                reference = self.model(batch)
                perturbed = self.model(batch, availability=sampled)
                bundle = self.pretraining_objective(perturbed, reference, batch, target, weight)
            finite_loss(bundle)
            self._backward(bundle.total, accumulation)
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                norm = self._step()
                bundle.values["gradient_norm"] = norm
            bundle.values["dropout_probability"] = probability
            ledger.update(bundle.values, batch.size)
        return ledger.distributed_means(self.device)

    def finetune_epoch(self, loader: DataLoader[PatientBatch]) -> dict[str, float]:
        self.model.train()
        ledger = MetricLedger()
        accumulation = self.config.optimizer.gradient_accumulation
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            with autocast_context(self.device, self.config.optimizer.precision):
                output = self.model(batch)
                bundle = self.finetuning_objective(output, batch)
            finite_loss(bundle)
            self._backward(bundle.total, accumulation)
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                norm = self._step()
                bundle.values["gradient_norm"] = norm
            ledger.update(bundle.values, batch.size)
        return ledger.distributed_means(self.device)

    @torch.no_grad()
    def validate(self, loader: DataLoader[PatientBatch]) -> dict[str, float]:
        self.model.eval()
        ledger = MetricLedger()
        for batch in loader:
            batch = batch.to(self.device)
            with autocast_context(self.device, self.config.optimizer.precision):
                output = self.model(batch)
                bundle = self.finetuning_objective(output, batch)
            ledger.update(bundle.values, batch.size)
        return ledger.distributed_means(self.device)

    def run_pretraining(
        self,
        train_loader: DataLoader[PatientBatch],
        validation_loader: DataLoader[PatientBatch],
        output: Path,
    ) -> None:
        scheduler = build_scheduler(self.optimizer, self.config.pretrain.epochs)
        best = float("inf")
        for epoch in range(1, self.config.pretrain.epochs + 1):
            started = time.monotonic()
            training = self.pretrain_epoch(train_loader, epoch)
            validation = self.validate(validation_loader)
            scheduler.step()
            metric = validation["total"]
            best = min(best, metric)
            state = EpochState("pretrain", epoch, self.global_step, best, self.config.seed)
            save_checkpoint(
                self.model,
                self.optimizer,
                scheduler,
                self.scaler,
                state,
                output / "pretrain-latest.pt",
            )
            if metric <= best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    state,
                    output / "pretrain-best.pt",
                )
            if is_primary():
                LOGGER.info(
                    "pretrain epoch=%d seconds=%.1f training=%s validation=%s",
                    epoch,
                    time.monotonic() - started,
                    training,
                    validation,
                )

    def run_finetuning(
        self,
        train_loader: DataLoader[PatientBatch],
        validation_loader: DataLoader[PatientBatch],
        output: Path,
    ) -> None:
        scheduler = build_scheduler(self.optimizer, self.config.finetune.epochs)
        best = float("inf")
        for epoch in range(1, self.config.finetune.epochs + 1):
            started = time.monotonic()
            training = self.finetune_epoch(train_loader)
            validation = self.validate(validation_loader)
            scheduler.step()
            metric = validation["total"]
            improved = metric < best
            best = min(best, metric)
            state = EpochState("finetune", epoch, self.global_step, best, self.config.seed)
            save_checkpoint(
                self.model,
                self.optimizer,
                scheduler,
                self.scaler,
                state,
                output / "finetune-latest.pt",
            )
            if improved:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    state,
                    output / "finetune-best.pt",
                )
            if is_primary():
                LOGGER.info(
                    "finetune epoch=%d seconds=%.1f training=%s validation=%s",
                    epoch,
                    time.monotonic() - started,
                    training,
                    validation,
                )


def write_run_record(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(values, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
