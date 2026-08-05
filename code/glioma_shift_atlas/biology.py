from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

VERHAAK_CLASSICAL = (
    "EGFR",
    "NES",
    "NOTCH3",
    "JAG1",
    "PDGFA",
    "FGFR3",
    "AKT2",
    "CDH4",
    "FGFRL1",
    "SOX9",
    "KLF4",
    "MEOX2",
    "PELI1",
    "GABRA5",
    "SLC1A5",
    "BCAT1",
    "MET",
    "PLAGL1",
    "TGFBR2",
    "RAB27A",
    "ARNT2",
    "PAX6",
    "SOX2",
    "OLIG1",
    "OLIG2",
)

VERHAAK_MESENCHYMAL = (
    "CHI3L1",
    "MET",
    "CD44",
    "MERTK",
    "TRADD",
    "RELB",
    "TNFRSF1A",
    "CASP1",
    "CASP4",
    "IL4R",
    "TLR2",
    "TLR4",
    "CCL2",
    "CXCL10",
    "CXCL12",
    "SERPINE1",
    "VIM",
    "LGALS3",
    "CTSC",
    "CTSB",
    "CTSL",
    "FCGR2A",
    "FCGR3A",
    "AIF1",
    "ITGAM",
    "STAT3",
    "CEBPB",
    "FOSL2",
    "RUNX1",
    "BCL3",
)

VERHAAK_PRONEURAL = (
    "PDGFRA",
    "OLIG2",
    "NKX2-2",
    "SOX10",
    "ERBB3",
    "DCX",
    "DLL3",
    "ASCL1",
    "TCF4",
    "BCAN",
    "GABRA2",
    "GABRB3",
    "GABRG1",
    "SLC1A1",
    "SLC1A3",
    "FABP7",
    "PLP1",
    "MBP",
    "MOG",
    "MAG",
    "CLDN11",
    "CNP",
    "MOBP",
    "SOX8",
    "SOX6",
)

VERHAAK_NEURAL = (
    "NEFL",
    "GABRA1",
    "SYT1",
    "SLC12A5",
    "SNAP25",
    "GRIN1",
    "GABRB2",
    "CAMK2A",
    "SYN1",
    "SYP",
    "RBFOX3",
    "MAP2",
    "TUBB3",
    "DLG4",
    "GRIA2",
    "GRIA3",
    "GAD1",
    "GAD2",
    "SLC17A7",
    "SLC32A1",
    "NRXN1",
    "NLGN1",
    "CNTNAP2",
    "KCNQ2",
    "SCN2A",
)

CELL_STATE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "astrocyte_like": (
        "GFAP",
        "AQP4",
        "ALDOC",
        "SLC1A3",
        "SLC1A2",
        "CLU",
        "APOE",
        "SPARCL1",
        "S100B",
        "GJA1",
        "FABP7",
        "HOPX",
        "SOX9",
        "VIM",
        "CD44",
    ),
    "oligodendrocyte_like": (
        "OLIG1",
        "OLIG2",
        "SOX10",
        "PDGFRA",
        "CSPG4",
        "PLP1",
        "MBP",
        "MOG",
        "MAG",
        "MOBP",
        "CNP",
        "CLDN11",
        "NKX2-2",
        "BCAS1",
        "GPR17",
    ),
    "neural_progenitor_like": (
        "SOX2",
        "SOX4",
        "SOX11",
        "DCX",
        "DLL3",
        "ASCL1",
        "HES6",
        "STMN2",
        "TUBB3",
        "CD24",
        "ELAVL4",
        "NEUROD1",
        "NEUROD2",
        "MAP2",
        "SEMA3C",
    ),
    "mesenchymal_hypoxic": (
        "CHI3L1",
        "CD44",
        "VIM",
        "SERPINE1",
        "CA9",
        "VEGFA",
        "LDHA",
        "SLC2A1",
        "ENO1",
        "BNIP3",
        "NDRG1",
        "ADM",
        "HILPDA",
        "ANGPTL4",
        "DDIT4",
    ),
    "microglia": (
        "P2RY12",
        "TMEM119",
        "CX3CR1",
        "GPR34",
        "SALL1",
        "OLFML3",
        "SIGLECH",
        "HEXB",
        "AIF1",
        "CSF1R",
        "C1QA",
        "C1QB",
        "C1QC",
        "TREM2",
        "TYROBP",
    ),
    "macrophage": (
        "CD68",
        "CD163",
        "MRC1",
        "MSR1",
        "FCGR3A",
        "FCGR2A",
        "LILRB1",
        "LILRB2",
        "CTSB",
        "CTSD",
        "CTSL",
        "APOC1",
        "LGALS3",
        "SPP1",
        "MARCO",
    ),
    "t_cell": (
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "TRBC1",
        "TRBC2",
        "CD247",
        "LCK",
        "IL7R",
        "CD4",
        "CD8A",
        "CD8B",
        "CCL5",
        "NKG7",
        "GZMK",
    ),
    "vasculature": (
        "PECAM1",
        "VWF",
        "KDR",
        "FLT1",
        "EMCN",
        "ENG",
        "RAMP2",
        "PLVAP",
        "ESAM",
        "CDH5",
        "CLDN5",
        "KLF2",
        "KLF4",
        "RGCC",
        "CA4",
    ),
    "pericyte": (
        "RGS5",
        "CSPG4",
        "PDGFRB",
        "MCAM",
        "NOTCH3",
        "COL4A1",
        "COL4A2",
        "DES",
        "ACTA2",
        "TAGLN",
        "MYL9",
        "ABCC9",
        "KCNJ8",
        "CPE",
        "NDUFA4L2",
    ),
}

VERHAAK_SIGNATURES = {
    "classical": VERHAAK_CLASSICAL,
    "mesenchymal": VERHAAK_MESENCHYMAL,
    "proneural": VERHAAK_PRONEURAL,
    "neural": VERHAAK_NEURAL,
}


def rank_expression(expression: pd.DataFrame) -> pd.DataFrame:
    return expression.rank(axis=0, method="average", pct=True)


def signature_score(expression: pd.DataFrame, genes: Sequence[str]) -> pd.Series:
    present = [gene for gene in genes if gene in expression.index]
    if not present:
        return pd.Series(np.zeros(expression.shape[1]), index=expression.columns)
    ranked = rank_expression(expression)
    return ranked.loc[present].mean(axis=0)


def signature_matrix(
    expression: pd.DataFrame,
    signatures: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    values = {name: signature_score(expression, genes) for name, genes in signatures.items()}
    return pd.DataFrame(values, index=expression.columns)


def soft_assignments(scores: pd.DataFrame, temperature: float = 0.1) -> pd.DataFrame:
    values = scores.to_numpy(dtype=np.float64)
    values = values / temperature
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return pd.DataFrame(probabilities, index=scores.index, columns=scores.columns)


def verhaak_assignments(expression: pd.DataFrame) -> pd.DataFrame:
    return soft_assignments(signature_matrix(expression, VERHAAK_SIGNATURES))


def cell_state_composition(expression: pd.DataFrame) -> pd.DataFrame:
    scores = signature_matrix(expression, CELL_STATE_SIGNATURES)
    positive = scores.clip(lower=0.0)
    denominator = positive.sum(axis=1).replace(0.0, 1.0)
    return positive.div(denominator, axis=0)


def compositional_shift(pre: pd.DataFrame, post: pd.DataFrame) -> pd.Series:
    shared = sorted(set(pre.columns) & set(post.columns))
    if not shared:
        raise ValueError("pre and post compositions must share states")
    pre_mean = pre[shared].mean(axis=0)
    post_mean = post[shared].mean(axis=0)
    return post_mean - pre_mean


def centered_log_ratio(composition: FloatArray, epsilon: float = 1e-6) -> FloatArray:
    clipped = np.clip(composition, epsilon, None)
    logarithm = np.log(clipped)
    return logarithm - logarithm.mean(axis=-1, keepdims=True)


def inverse_centered_log_ratio(value: FloatArray) -> FloatArray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def js_divergence(first: FloatArray, second: FloatArray) -> float:
    first = np.clip(first, 1e-12, None)
    second = np.clip(second, 1e-12, None)
    first = first / first.sum()
    second = second / second.sum()
    midpoint = 0.5 * (first + second)
    left = stats.entropy(first, midpoint)
    right = stats.entropy(second, midpoint)
    return float(0.5 * (left + right))


def shannon_diversity(composition: FloatArray) -> float:
    probability = np.clip(composition, 1e-12, None)
    probability = probability / probability.sum()
    return float(stats.entropy(probability))


def simpson_diversity(composition: FloatArray) -> float:
    probability = np.clip(composition, 0.0, None)
    probability = probability / probability.sum()
    return float(1.0 - np.sum(probability**2))


@dataclass(frozen=True)
class NicheSummary:
    dominant_state: str
    dominant_fraction: float
    shannon_diversity: float
    simpson_diversity: float
    hypoxia_score: float
    immune_fraction: float
    vascular_fraction: float


def niche_summary(composition: pd.Series, scores: pd.Series) -> NicheSummary:
    values = composition.to_numpy(dtype=np.float64)
    dominant = str(composition.index[int(np.argmax(values))])
    immune = sum(float(composition.get(name, 0.0)) for name in ("microglia", "macrophage", "t_cell"))
    vascular = sum(float(composition.get(name, 0.0)) for name in ("vasculature", "pericyte"))
    return NicheSummary(
        dominant_state=dominant,
        dominant_fraction=float(np.max(values)),
        shannon_diversity=shannon_diversity(values),
        simpson_diversity=simpson_diversity(values),
        hypoxia_score=float(scores.get("mesenchymal_hypoxic", 0.0)),
        immune_fraction=immune,
        vascular_fraction=vascular,
    )
