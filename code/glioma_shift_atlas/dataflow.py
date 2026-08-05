from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from glioma_shift_atlas.contracts import PatientBatch, PatientRecord, SplitAssignment


def optional_path(value: str) -> Path | None:
    stripped = value.strip()
    return Path(stripped) if stripped else None


def optional_integer(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped else None


def parse_subtype_scores(row: dict[str, str]) -> tuple[float, float, float, float]:
    values = tuple(float(row[f"subtype_{index}"]) for index in range(4))
    if len(values) != 4:
        raise ValueError("four subtype scores are required")
    return values


def load_manifest(path: Path) -> list[PatientRecord]:
    records: list[PatientRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                PatientRecord(
                    patient_key=row["patient_key"],
                    cohort=row["cohort"],
                    histology_path=optional_path(row.get("histology_path", "")),
                    spatial_path=optional_path(row.get("spatial_path", "")),
                    single_cell_path=optional_path(row.get("single_cell_path", "")),
                    bulk_path=optional_path(row.get("bulk_path", "")),
                    clinical_path=optional_path(row.get("clinical_path", "")),
                    overall_survival_months=float(row["overall_survival_months"]),
                    event=int(row["event"]),
                    therapy_response=optional_integer(row.get("therapy_response", "")),
                    idh_status=optional_integer(row.get("idh_status", "")),
                    mgmt_status=optional_integer(row.get("mgmt_status", "")),
                    subtype_scores=parse_subtype_scores(row),
                    shift_path=optional_path(row.get("shift_path", "")),
                )
            )
    validate_records(records)
    return records


def validate_records(records: Sequence[PatientRecord]) -> None:
    if not records:
        raise ValueError("manifest is empty")
    keys = [record.patient_key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("patient keys must be unique")
    for record in records:
        if record.overall_survival_months < 0:
            raise ValueError("survival time must be nonnegative")
        if record.event not in (0, 1):
            raise ValueError("event must be binary")
        if record.therapy_response not in (None, 0, 1):
            raise ValueError("therapy response must be binary")
        if record.idh_status not in (None, 0, 1):
            raise ValueError("IDH status must be binary")
        if abs(sum(record.subtype_scores) - 1.0) > 1e-4:
            raise ValueError("subtype scores must sum to one")
        paths = (
            record.histology_path,
            record.spatial_path,
            record.single_cell_path,
            record.bulk_path,
            record.clinical_path,
        )
        if all(path is None for path in paths):
            raise ValueError("each patient requires at least one modality")


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(path: Path) -> str:
    frame = pd.read_csv(path, dtype=str).fillna("")
    columns = sorted(frame.columns.tolist())
    frame = frame[columns].sort_values(columns).reset_index(drop=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratified_assignments(
    records: Sequence[PatientRecord],
    folds: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> list[SplitAssignment]:
    generator = np.random.default_rng(seed)
    grouped: dict[tuple[str, int, int], list[PatientRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.cohort, record.event, record.idh_status or -1)].append(record)
    assignments: list[SplitAssignment] = []
    for group in grouped.values():
        order = generator.permutation(len(group))
        for position, index in enumerate(order.tolist()):
            record = group[index]
            fraction = position / max(1, len(group))
            if fraction < train_fraction:
                partition = "train"
            elif fraction < train_fraction + validation_fraction:
                partition = "validation"
            else:
                partition = "test"
            assignments.append(
                SplitAssignment(
                    patient_key=record.patient_key,
                    cohort=record.cohort,
                    fold=position % folds,
                    partition=partition,
                )
            )
    return assignments


def save_assignments(assignments: Sequence[SplitAssignment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "patient_key": item.patient_key,
                "cohort": item.cohort,
                "fold": item.fold,
                "partition": item.partition,
            }
            for item in assignments
        ]
    )
    frame.to_csv(path, index=False)


def load_numpy(path: Path) -> NDArray[np.float32]:
    value = np.load(path, allow_pickle=False)
    return np.asarray(value, dtype=np.float32)


def load_hdf5(path: Path, key: str) -> NDArray[np.float32]:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key], dtype=np.float32)


def load_expression(path: Path, key: str = "expression") -> Tensor:
    if path.suffix == ".npy":
        value = load_numpy(path)
    elif path.suffix in (".h5", ".hdf5"):
        value = load_hdf5(path, key)
    elif path.suffix in (".csv", ".tsv"):
        separator = "\t" if path.suffix == ".tsv" else ","
        value = pd.read_csv(path, sep=separator, index_col=0).to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"unsupported expression format: {path.suffix}")
    return torch.from_numpy(value)


def load_spatial(path: Path) -> tuple[Tensor, Tensor]:
    if path.suffix not in (".h5", ".hdf5"):
        raise ValueError("spatial inputs must be HDF5")
    expression = load_hdf5(path, "expression")
    coordinates = load_hdf5(path, "coordinates")
    if expression.shape[0] != coordinates.shape[0]:
        raise ValueError("spatial expression and coordinates must align")
    if coordinates.shape[1] != 2:
        raise ValueError("spatial coordinates require two dimensions")
    return torch.from_numpy(expression), torch.from_numpy(coordinates)


def load_clinical(path: Path, dimension: int = 12) -> Tensor:
    frame = pd.read_csv(path)
    values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32).reshape(-1)
    if values.shape[0] != dimension:
        raise ValueError(f"clinical vector must have {dimension} values")
    return torch.from_numpy(values)


def load_shift(path: Path) -> Tensor:
    value = load_expression(path).reshape(-1)
    return value


def image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def extract_grid_patches(
    path: Path,
    patch_size: int,
    maximum_patches: int,
    tissue_threshold: float = 0.75,
) -> Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    patches: list[Tensor] = []
    for top in range(0, max(1, height - patch_size + 1), patch_size):
        for left in range(0, max(1, width - patch_size + 1), patch_size):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            tensor = image_to_tensor(patch)
            brightness = tensor.mean(dim=0)
            tissue_fraction = float((brightness < 0.9).float().mean())
            if tissue_fraction >= tissue_threshold:
                patches.append(tensor)
            if len(patches) >= maximum_patches:
                return torch.stack(patches)
    if not patches:
        resized = image.resize((patch_size, patch_size))
        patches.append(image_to_tensor(resized))
    return torch.stack(patches)


@dataclass
class LoadedPatient:
    record: PatientRecord
    histology: Tensor | None
    spatial_expression: Tensor | None
    spatial_coordinates: Tensor | None
    single_cell_expression: Tensor | None
    bulk_expression: Tensor | None
    clinical: Tensor | None
    shift: Tensor | None


class GliomaDataset(Dataset[LoadedPatient]):
    def __init__(
        self,
        records: Sequence[PatientRecord],
        patch_size: int,
        maximum_patches: int = 200,
        transform: Callable[[LoadedPatient], LoadedPatient] | None = None,
    ) -> None:
        self.records = list(records)
        self.patch_size = patch_size
        self.maximum_patches = maximum_patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LoadedPatient:
        record = self.records[index]
        histology = None
        if record.histology_path is not None:
            histology = extract_grid_patches(
                record.histology_path,
                self.patch_size,
                self.maximum_patches,
            )
        spatial_expression = None
        spatial_coordinates = None
        if record.spatial_path is not None:
            spatial_expression, spatial_coordinates = load_spatial(record.spatial_path)
        single_cell = (
            load_expression(record.single_cell_path)
            if record.single_cell_path is not None
            else None
        )
        bulk = load_expression(record.bulk_path).reshape(-1) if record.bulk_path is not None else None
        clinical = load_clinical(record.clinical_path) if record.clinical_path is not None else None
        shift = load_shift(record.shift_path) if record.shift_path is not None else None
        loaded = LoadedPatient(
            record=record,
            histology=histology,
            spatial_expression=spatial_expression,
            spatial_coordinates=spatial_coordinates,
            single_cell_expression=single_cell,
            bulk_expression=bulk,
            clinical=clinical,
            shift=shift,
        )
        return self.transform(loaded) if self.transform is not None else loaded


def pad_sequence(values: Sequence[Tensor | None]) -> tuple[Tensor | None, Tensor | None]:
    present = [value for value in values if value is not None]
    if not present:
        return None, None
    maximum = max(value.shape[0] for value in present)
    trailing = present[0].shape[1:]
    output = present[0].new_zeros((len(values), maximum, *trailing))
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        if value is None:
            continue
        length = value.shape[0]
        output[index, :length] = value
        mask[index, :length] = True
    return output, mask


def stack_optional(values: Sequence[Tensor | None]) -> Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    reference = present[0]
    output = reference.new_zeros((len(values), *reference.shape))
    for index, value in enumerate(values):
        if value is not None:
            if value.shape != reference.shape:
                raise ValueError("fixed-size modalities must share shape")
            output[index] = value
    return output


def collate_patients(items: Sequence[LoadedPatient]) -> PatientBatch:
    histology, histology_mask = pad_sequence([item.histology for item in items])
    spatial_expression, spatial_mask = pad_sequence([item.spatial_expression for item in items])
    spatial_coordinates, _ = pad_sequence([item.spatial_coordinates for item in items])
    single_cell, single_cell_mask = pad_sequence([item.single_cell_expression for item in items])
    bulk = stack_optional([item.bulk_expression for item in items])
    clinical = stack_optional([item.clinical for item in items])
    shift = stack_optional([item.shift for item in items])
    if shift is None:
        shift = torch.zeros(len(items), 1)
    modality_mask = torch.tensor(
        [
            [
                item.histology is not None,
                item.spatial_expression is not None,
                item.single_cell_expression is not None,
                item.bulk_expression is not None,
                item.clinical is not None,
            ]
            for item in items
        ],
        dtype=torch.float32,
    )
    return PatientBatch(
        patient_keys=[item.record.patient_key for item in items],
        cohorts=[item.record.cohort for item in items],
        histology=histology,
        histology_mask=histology_mask,
        spatial_expression=spatial_expression,
        spatial_coordinates=spatial_coordinates,
        spatial_mask=spatial_mask,
        single_cell_expression=single_cell,
        single_cell_mask=single_cell_mask,
        bulk_expression=bulk,
        clinical=clinical,
        modality_mask=modality_mask,
        survival_time=torch.tensor([item.record.overall_survival_months for item in items]),
        survival_event=torch.tensor([item.record.event for item in items]),
        therapy_response=torch.tensor([item.record.therapy_response or 0 for item in items]),
        therapy_observed=torch.tensor([item.record.therapy_response is not None for item in items]),
        idh_status=torch.tensor([item.record.idh_status or 0 for item in items]),
        idh_observed=torch.tensor([item.record.idh_status is not None for item in items]),
        subtype_scores=torch.tensor([item.record.subtype_scores for item in items]),
        compositional_shift=shift,
        shift_observed=torch.tensor([item.shift is not None for item in items]),
    )


class CohortBalancedSampler(Sampler[int]):
    def __init__(self, records: Sequence[PatientRecord], seed: int, epoch_size: int | None = None) -> None:
        self.records = list(records)
        self.seed = seed
        self.epoch = 0
        self.epoch_size = epoch_size or len(records)
        self.groups: dict[str, NDArray[np.int64]] = {}
        for cohort in sorted({record.cohort for record in records}):
            self.groups[cohort] = np.asarray(
                [index for index, record in enumerate(records) if record.cohort == cohort],
                dtype=np.int64,
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        cohorts = sorted(self.groups)
        for position in range(self.epoch_size):
            cohort = cohorts[position % len(cohorts)]
            candidates = self.groups[cohort]
            yield int(generator.choice(candidates))


def write_data_registry(manifest: Path, records: Sequence[PatientRecord], destination: Path) -> None:
    cohort_counts: dict[str, int] = defaultdict(int)
    modality_counts: dict[str, int] = defaultdict(int)
    for record in records:
        cohort_counts[record.cohort] += 1
        modality_counts["histology"] += int(record.histology_path is not None)
        modality_counts["spatial"] += int(record.spatial_path is not None)
        modality_counts["single_cell"] += int(record.single_cell_path is not None)
        modality_counts["bulk"] += int(record.bulk_path is not None)
        modality_counts["clinical"] += int(record.clinical_path is not None)
    payload = {
        "manifest_sha256": manifest_fingerprint(manifest),
        "patients": len(records),
        "cohorts": dict(cohort_counts),
        "modalities": dict(modality_counts),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
