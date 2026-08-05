from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from sklearn.preprocessing import QuantileTransformer


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GeneIndex:
    symbols: tuple[str, ...]
    positions: Mapping[str, int]

    @classmethod
    def from_symbols(cls, symbols: Sequence[str]) -> GeneIndex:
        normalized = tuple(normalize_gene_symbol(symbol) for symbol in symbols)
        positions = {symbol: index for index, symbol in enumerate(normalized)}
        return cls(normalized, positions)


@dataclass(frozen=True)
class ExpressionSummary:
    samples: int
    genes: int
    nonzero_fraction: float
    median_library_size: float
    minimum_library_size: float
    maximum_library_size: float


@dataclass(frozen=True)
class SpatialSummary:
    spots: int
    genes: int
    width: float
    height: float
    median_neighbor_distance: float


def normalize_gene_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value and value.rsplit(".", 1)[-1].isdigit():
        value = value.rsplit(".", 1)[0]
    return value


def make_unique_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    counts: dict[str, int] = defaultdict(int)
    output: list[str] = []
    for symbol in symbols:
        normalized = normalize_gene_symbol(symbol)
        count = counts[normalized]
        output.append(normalized if count == 0 else f"{normalized}-{count}")
        counts[normalized] += 1
    return tuple(output)


def library_size_normalize(matrix: FloatArray, target: float = 1e4) -> FloatArray:
    totals = matrix.sum(axis=1, keepdims=True)
    scale = target / np.clip(totals, 1.0, None)
    return matrix * scale


def log_normalize(matrix: FloatArray, target: float = 1e4) -> FloatArray:
    return np.log1p(library_size_normalize(matrix, target))


def zscore_genes(matrix: FloatArray, epsilon: float = 1e-8) -> FloatArray:
    mean = matrix.mean(axis=0, keepdims=True)
    deviation = matrix.std(axis=0, keepdims=True)
    return (matrix - mean) / np.clip(deviation, epsilon, None)


def clip_expression(matrix: FloatArray, limit: float = 10.0) -> FloatArray:
    return np.clip(matrix, -limit, limit)


def variance_filter(matrix: FloatArray, genes: int) -> IntArray:
    variance = matrix.var(axis=0)
    count = min(genes, variance.shape[0])
    return np.argsort(variance)[-count:].astype(np.int64)


def detection_filter(matrix: FloatArray, minimum_fraction: float) -> IntArray:
    detected = np.mean(matrix > 0, axis=0)
    return np.nonzero(detected >= minimum_fraction)[0].astype(np.int64)


def mitochondrial_fraction(matrix: FloatArray, symbols: Sequence[str]) -> FloatArray:
    selected = np.asarray([symbol.startswith("MT-") for symbol in symbols], dtype=bool)
    total = matrix.sum(axis=1)
    mitochondrial = matrix[:, selected].sum(axis=1) if np.any(selected) else np.zeros(matrix.shape[0])
    return mitochondrial / np.clip(total, 1.0, None)


def ribosomal_fraction(matrix: FloatArray, symbols: Sequence[str]) -> FloatArray:
    selected = np.asarray(
        [symbol.startswith("RPL") or symbol.startswith("RPS") for symbol in symbols],
        dtype=bool,
    )
    total = matrix.sum(axis=1)
    ribosomal = matrix[:, selected].sum(axis=1) if np.any(selected) else np.zeros(matrix.shape[0])
    return ribosomal / np.clip(total, 1.0, None)


def cell_quality_mask(
    matrix: FloatArray,
    symbols: Sequence[str],
    minimum_genes: int = 200,
    maximum_genes: int = 8000,
    maximum_mitochondrial: float = 0.25,
) -> NDArray[np.bool_]:
    detected = np.sum(matrix > 0, axis=1)
    mitochondrial = mitochondrial_fraction(matrix, symbols)
    return (
        (detected >= minimum_genes)
        & (detected <= maximum_genes)
        & (mitochondrial <= maximum_mitochondrial)
    )


def align_expression(
    matrix: FloatArray,
    source_symbols: Sequence[str],
    target_symbols: Sequence[str],
) -> FloatArray:
    source = GeneIndex.from_symbols(source_symbols)
    output = np.zeros((matrix.shape[0], len(target_symbols)), dtype=np.float64)
    for target_index, symbol in enumerate(target_symbols):
        source_index = source.positions.get(normalize_gene_symbol(symbol))
        if source_index is not None:
            output[:, target_index] = matrix[:, source_index]
    return output


def quantile_map(source: FloatArray, reference: FloatArray, seed: int = 42) -> FloatArray:
    combined = np.concatenate((reference, source), axis=0)
    transformer = QuantileTransformer(
        n_quantiles=min(1000, combined.shape[0]),
        output_distribution="normal",
        random_state=seed,
        copy=True,
    )
    transformed = transformer.fit_transform(combined)
    return np.asarray(transformed[reference.shape[0] :], dtype=np.float64)


def expression_summary(matrix: FloatArray) -> ExpressionSummary:
    libraries = matrix.sum(axis=1)
    return ExpressionSummary(
        samples=matrix.shape[0],
        genes=matrix.shape[1],
        nonzero_fraction=float(np.mean(matrix != 0)),
        median_library_size=float(np.median(libraries)),
        minimum_library_size=float(np.min(libraries)),
        maximum_library_size=float(np.max(libraries)),
    )


def pairwise_distances(coordinates: FloatArray) -> FloatArray:
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(difference**2, axis=-1))


def radius_neighbors(coordinates: FloatArray, radius: float) -> list[IntArray]:
    distances = pairwise_distances(coordinates)
    output: list[IntArray] = []
    for index in range(coordinates.shape[0]):
        selected = np.nonzero((distances[index] <= radius) & (distances[index] > 0))[0]
        output.append(selected.astype(np.int64))
    return output


def knn_neighbors(coordinates: FloatArray, neighbors: int) -> list[IntArray]:
    distances = pairwise_distances(coordinates)
    np.fill_diagonal(distances, np.inf)
    count = min(neighbors, max(0, coordinates.shape[0] - 1))
    return [np.argsort(row)[:count].astype(np.int64) for row in distances]


def niche_aggregate(matrix: FloatArray, neighborhoods: Sequence[IntArray]) -> FloatArray:
    output = np.zeros_like(matrix)
    for index, neighbors in enumerate(neighborhoods):
        members = np.concatenate((np.asarray([index], dtype=np.int64), neighbors))
        output[index] = matrix[members].mean(axis=0)
    return output


def spatial_summary(matrix: FloatArray, coordinates: FloatArray) -> SpatialSummary:
    distances = pairwise_distances(coordinates)
    positive = distances[distances > 0]
    return SpatialSummary(
        spots=matrix.shape[0],
        genes=matrix.shape[1],
        width=float(np.ptp(coordinates[:, 0])),
        height=float(np.ptp(coordinates[:, 1])),
        median_neighbor_distance=float(np.median(positive)) if positive.shape[0] else 0.0,
    )


def save_expression_hdf5(
    path: Path,
    expression: FloatArray,
    symbols: Sequence[str],
    coordinates: FloatArray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("expression", data=expression.astype(np.float32), compression="gzip")
        encoded = np.asarray([symbol.encode("utf-8") for symbol in symbols])
        handle.create_dataset("genes", data=encoded)
        if coordinates is not None:
            handle.create_dataset("coordinates", data=coordinates.astype(np.float32))


def load_mtx_bundle(matrix_path: Path, genes_path: Path) -> tuple[FloatArray, tuple[str, ...]]:
    matrix = sparse.load_npz(matrix_path).toarray().astype(np.float64)
    genes = tuple(pd.read_csv(genes_path, header=None, sep="\t").iloc[:, 0].astype(str))
    if matrix.shape[1] != len(genes):
        if matrix.shape[0] == len(genes):
            matrix = matrix.T
        else:
            raise ValueError("matrix and genes are inconsistent")
    return matrix, genes


def prepare_expression(
    matrix: FloatArray,
    symbols: Sequence[str],
    target_symbols: Sequence[str],
    variable_genes: int | None = None,
) -> tuple[FloatArray, tuple[str, ...]]:
    aligned = align_expression(matrix, symbols, target_symbols)
    normalized = clip_expression(zscore_genes(log_normalize(aligned)))
    selected_symbols = tuple(target_symbols)
    if variable_genes is not None and variable_genes < normalized.shape[1]:
        indices = variance_filter(normalized, variable_genes)
        normalized = normalized[:, indices]
        selected_symbols = tuple(target_symbols[index] for index in indices.tolist())
    return normalized, selected_symbols


def directory_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def write_preparation_record(
    path: Path,
    expression: ExpressionSummary,
    spatial: SpatialSummary | None,
    sources: Sequence[Path],
) -> None:
    payload = {
        "expression": asdict(expression),
        "spatial": asdict(spatial) if spatial is not None else None,
        "source_digest": directory_digest(sources),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
