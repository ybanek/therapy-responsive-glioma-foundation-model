from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn import metrics

from glioma_shift_atlas.contracts import MetricEstimate, SubgroupEstimate

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def as_float(values: Sequence[float] | NDArray[np.floating]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def as_int(values: Sequence[int] | NDArray[np.integer]) -> IntArray:
    return np.asarray(values, dtype=np.int64)


def concordance_index(time: FloatArray, risk: FloatArray, event: IntArray) -> float:
    concordant = 0.0
    comparable = 0.0
    size = time.shape[0]
    for left in range(size):
        for right in range(left + 1, size):
            if time[left] == time[right]:
                continue
            if time[left] < time[right] and event[left] == 1:
                comparable += 1.0
                if risk[left] > risk[right]:
                    concordant += 1.0
                elif risk[left] == risk[right]:
                    concordant += 0.5
            elif time[right] < time[left] and event[right] == 1:
                comparable += 1.0
                if risk[right] > risk[left]:
                    concordant += 1.0
                elif risk[right] == risk[left]:
                    concordant += 0.5
    return concordant / comparable if comparable > 0 else float("nan")


def binary_auc(target: IntArray, score: FloatArray) -> float:
    if np.unique(target).shape[0] < 2:
        return float("nan")
    return float(metrics.roc_auc_score(target, score))


def average_precision(target: IntArray, score: FloatArray) -> float:
    if np.unique(target).shape[0] < 2:
        return float("nan")
    return float(metrics.average_precision_score(target, score))


def macro_f1(target: IntArray, prediction: IntArray) -> float:
    return float(metrics.f1_score(target, prediction, average="macro", zero_division=0))


def weighted_f1(target: IntArray, prediction: IntArray) -> float:
    return float(metrics.f1_score(target, prediction, average="weighted", zero_division=0))


def cohen_kappa(target: IntArray, prediction: IntArray) -> float:
    return float(metrics.cohen_kappa_score(target, prediction))


def balanced_accuracy(target: IntArray, prediction: IntArray) -> float:
    return float(metrics.balanced_accuracy_score(target, prediction))


def sensitivity(target: IntArray, prediction: IntArray) -> float:
    true_positive = np.sum((target == 1) & (prediction == 1))
    false_negative = np.sum((target == 1) & (prediction == 0))
    denominator = true_positive + false_negative
    return float(true_positive / denominator) if denominator else float("nan")


def specificity(target: IntArray, prediction: IntArray) -> float:
    true_negative = np.sum((target == 0) & (prediction == 0))
    false_positive = np.sum((target == 0) & (prediction == 1))
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else float("nan")


def positive_predictive_value(target: IntArray, prediction: IntArray) -> float:
    true_positive = np.sum((target == 1) & (prediction == 1))
    false_positive = np.sum((target == 0) & (prediction == 1))
    denominator = true_positive + false_positive
    return float(true_positive / denominator) if denominator else float("nan")


def negative_predictive_value(target: IntArray, prediction: IntArray) -> float:
    true_negative = np.sum((target == 0) & (prediction == 0))
    false_negative = np.sum((target == 1) & (prediction == 0))
    denominator = true_negative + false_negative
    return float(true_negative / denominator) if denominator else float("nan")


def brier_score(target: IntArray, probability: FloatArray) -> float:
    return float(np.mean((probability - target) ** 2))


def expected_calibration_error(
    target: IntArray,
    probability: FloatArray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = target.shape[0]
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        count = int(np.sum(selected))
        if count == 0:
            continue
        accuracy = float(np.mean(target[selected]))
        confidence = float(np.mean(probability[selected]))
        value += count / total * abs(accuracy - confidence)
    return value


def adaptive_calibration_error(
    target: IntArray,
    probability: FloatArray,
    bins: int = 10,
) -> float:
    order = np.argsort(probability)
    groups = np.array_split(order, bins)
    value = 0.0
    for group in groups:
        if group.shape[0] == 0:
            continue
        accuracy = float(np.mean(target[group]))
        confidence = float(np.mean(probability[group]))
        value += group.shape[0] / target.shape[0] * abs(accuracy - confidence)
    return value


def calibration_intercept_slope(target: IntArray, probability: FloatArray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(logits.shape[0]), logits))
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        linear = design @ coefficients
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        working = linear + (target - fitted) / weights
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_working = working * np.sqrt(weights)
        updated, _, _, _ = np.linalg.lstsq(weighted_design, weighted_working, rcond=None)
        if np.max(np.abs(updated - coefficients)) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return float(coefficients[0]), float(coefficients[1])


def hosmer_lemeshow(
    target: IntArray,
    probability: FloatArray,
    groups: int = 10,
) -> tuple[float, float]:
    order = np.argsort(probability)
    partitions = np.array_split(order, groups)
    statistic = 0.0
    populated = 0
    for partition in partitions:
        if partition.shape[0] == 0:
            continue
        observed = float(np.sum(target[partition]))
        expected = float(np.sum(probability[partition]))
        count = partition.shape[0]
        denominator = expected * (1.0 - expected / count)
        if denominator > 0:
            statistic += (observed - expected) ** 2 / denominator
            populated += 1
    degrees = max(1, populated - 2)
    return statistic, float(stats.chi2.sf(statistic, degrees))


def kaplan_meier(time: FloatArray, event: IntArray) -> tuple[FloatArray, FloatArray]:
    unique = np.unique(time[event == 1])
    survival = 1.0
    times = [0.0]
    values = [1.0]
    for point in np.sort(unique):
        at_risk = np.sum(time >= point)
        events = np.sum((time == point) & (event == 1))
        if at_risk > 0:
            survival *= 1.0 - events / at_risk
        times.append(float(point))
        values.append(survival)
    return as_float(times), as_float(values)


def survival_at(times: FloatArray, survival: FloatArray, horizon: float) -> float:
    indices = np.nonzero(times <= horizon)[0]
    return float(survival[indices[-1]]) if indices.shape[0] else 1.0


def logrank_test(
    time: FloatArray,
    event: IntArray,
    group: IntArray,
) -> tuple[float, float]:
    unique = np.unique(time[event == 1])
    observed_first = 0.0
    expected_first = 0.0
    variance = 0.0
    for point in unique:
        risk_first = np.sum((time >= point) & (group == 0))
        risk_second = np.sum((time >= point) & (group == 1))
        events_first = np.sum((time == point) & (event == 1) & (group == 0))
        events_total = np.sum((time == point) & (event == 1))
        risk_total = risk_first + risk_second
        if risk_total <= 1:
            continue
        expected = events_total * risk_first / risk_total
        component = (
            risk_first
            * risk_second
            * events_total
            * (risk_total - events_total)
            / (risk_total**2 * (risk_total - 1))
        )
        observed_first += events_first
        expected_first += expected
        variance += component
    statistic = (observed_first - expected_first) ** 2 / variance if variance > 0 else 0.0
    return statistic, float(stats.chi2.sf(statistic, 1))


def hazard_ratio_binary(
    time: FloatArray,
    event: IntArray,
    group: IntArray,
    iterations: int = 100,
) -> tuple[float, float, float, float]:
    coefficient = 0.0
    information = 0.0
    event_times = np.unique(time[event == 1])
    for _ in range(iterations):
        score = 0.0
        information = 0.0
        for point in event_times:
            risk = time >= point
            deaths = (time == point) & (event == 1)
            death_count = int(np.sum(deaths))
            if death_count == 0:
                continue
            weights = np.exp(np.clip(coefficient * group[risk], -30.0, 30.0))
            covariates = group[risk].astype(np.float64)
            weighted_mean = float(np.sum(weights * covariates) / np.sum(weights))
            weighted_second = float(np.sum(weights * covariates**2) / np.sum(weights))
            score += float(np.sum(group[deaths])) - death_count * weighted_mean
            information += death_count * (weighted_second - weighted_mean**2)
        if information <= 0:
            break
        step = score / information
        coefficient += step
        if abs(step) < 1e-8:
            break
    standard_error = math.sqrt(1.0 / information) if information > 0 else float("inf")
    hazard = math.exp(coefficient)
    lower = math.exp(coefficient - 1.96 * standard_error)
    upper = math.exp(coefficient + 1.96 * standard_error)
    z = coefficient / standard_error if standard_error > 0 else 0.0
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    return hazard, lower, upper, p_value


def bootstrap_indices(size: int, samples: int, seed: int) -> Iterable[IntArray]:
    generator = np.random.default_rng(seed)
    for _ in range(samples):
        yield generator.integers(0, size, size=size, dtype=np.int64)


def bootstrap_estimate(
    name: str,
    function: Callable[[IntArray], float],
    size: int,
    samples: int,
    confidence: float,
    seed: int,
) -> MetricEstimate:
    point = function(np.arange(size, dtype=np.int64))
    values = np.asarray(
        [function(indices) for indices in bootstrap_indices(size, samples, seed)],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    alpha = 1.0 - confidence
    lower = float(np.quantile(values, alpha / 2.0)) if values.shape[0] else float("nan")
    upper = float(np.quantile(values, 1.0 - alpha / 2.0)) if values.shape[0] else float("nan")
    deviation = float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0
    return MetricEstimate(name, point, lower, upper, deviation, size)


def bootstrap_auc(
    target: IntArray,
    score: FloatArray,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> MetricEstimate:
    return bootstrap_estimate(
        "auc",
        lambda indices: binary_auc(target[indices], score[indices]),
        target.shape[0],
        samples,
        confidence,
        seed,
    )


def bootstrap_concordance(
    time: FloatArray,
    risk: FloatArray,
    event: IntArray,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> MetricEstimate:
    return bootstrap_estimate(
        "c_index",
        lambda indices: concordance_index(time[indices], risk[indices], event[indices]),
        time.shape[0],
        samples,
        confidence,
        seed,
    )


def bootstrap_kappa(
    target: IntArray,
    prediction: IntArray,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> MetricEstimate:
    return bootstrap_estimate(
        "cohen_kappa",
        lambda indices: cohen_kappa(target[indices], prediction[indices]),
        target.shape[0],
        samples,
        confidence,
        seed,
    )


def permutation_auc_difference(
    target: IntArray,
    first: FloatArray,
    second: FloatArray,
    permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    observed = binary_auc(target, first) - binary_auc(target, second)
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        exchange = generator.random(target.shape[0]) < 0.5
        permuted_first = np.where(exchange, second, first)
        permuted_second = np.where(exchange, first, second)
        difference = binary_auc(target, permuted_first) - binary_auc(target, permuted_second)
        if abs(difference) >= abs(observed):
            exceed += 1
    return observed, (exceed + 1.0) / (permutations + 1.0)


def net_reclassification_improvement(
    target: IntArray,
    proposed: FloatArray,
    reference: FloatArray,
    thresholds: Sequence[float],
) -> float:
    boundaries = np.asarray(thresholds, dtype=np.float64)
    proposed_class = np.digitize(proposed, boundaries)
    reference_class = np.digitize(reference, boundaries)
    cases = target == 1
    controls = target == 0
    case_up = np.mean(proposed_class[cases] > reference_class[cases]) if np.any(cases) else 0.0
    case_down = np.mean(proposed_class[cases] < reference_class[cases]) if np.any(cases) else 0.0
    control_down = np.mean(proposed_class[controls] < reference_class[controls]) if np.any(controls) else 0.0
    control_up = np.mean(proposed_class[controls] > reference_class[controls]) if np.any(controls) else 0.0
    return float(case_up - case_down + control_down - control_up)


def integrated_discrimination_improvement(
    target: IntArray,
    proposed: FloatArray,
    reference: FloatArray,
) -> float:
    cases = target == 1
    controls = target == 0
    proposed_discrimination = np.mean(proposed[cases]) - np.mean(proposed[controls])
    reference_discrimination = np.mean(reference[cases]) - np.mean(reference[controls])
    return float(proposed_discrimination - reference_discrimination)


def cochran_q(estimates: FloatArray, variances: FloatArray) -> tuple[float, float, float, float]:
    weights = 1.0 / np.clip(variances, 1e-12, None)
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    q = float(np.sum(weights * (estimates - pooled) ** 2))
    degrees = estimates.shape[0] - 1
    p_value = float(stats.chi2.sf(q, degrees)) if degrees > 0 else 1.0
    i_squared = max(0.0, (q - degrees) / q) * 100.0 if q > 0 else 0.0
    correction = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau_squared = max(0.0, (q - degrees) / correction) if correction > 0 else 0.0
    return q, p_value, i_squared, float(tau_squared)


def random_effects_meta_analysis(
    estimates: FloatArray,
    variances: FloatArray,
) -> tuple[float, float, float]:
    _, _, _, tau_squared = cochran_q(estimates, variances)
    weights = 1.0 / (variances + tau_squared)
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    standard_error = math.sqrt(1.0 / np.sum(weights))
    return pooled, pooled - 1.96 * standard_error, pooled + 1.96 * standard_error


def subgroup_estimates(
    metric_name: str,
    target: IntArray,
    score: FloatArray,
    subgroup: Sequence[str],
    samples: int,
    confidence: float,
    seed: int,
) -> list[SubgroupEstimate]:
    labels = np.asarray(subgroup, dtype=str)
    output: list[SubgroupEstimate] = []
    for level in sorted(np.unique(labels).tolist()):
        selected = labels == level
        estimate = bootstrap_auc(
            target[selected],
            score[selected],
            samples=samples,
            confidence=confidence,
            seed=seed,
        )
        output.append(
            SubgroupEstimate(
                metric=metric_name,
                subgroup="subgroup",
                level=level,
                value=estimate.value,
                lower=estimate.lower,
                upper=estimate.upper,
                sample_count=estimate.sample_count,
            )
        )
    return output


@dataclass(frozen=True)
class ClassificationReport:
    auc: float
    average_precision: float
    brier: float
    ece: float
    ace: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float
    calibration_intercept: float
    calibration_slope: float
    hosmer_lemeshow_statistic: float
    hosmer_lemeshow_p_value: float


def classification_report(
    target: IntArray,
    probability: FloatArray,
    threshold: float = 0.5,
) -> ClassificationReport:
    prediction = (probability >= threshold).astype(np.int64)
    intercept, slope = calibration_intercept_slope(target, probability)
    hl_statistic, hl_p = hosmer_lemeshow(target, probability)
    return ClassificationReport(
        auc=binary_auc(target, probability),
        average_precision=average_precision(target, probability),
        brier=brier_score(target, probability),
        ece=expected_calibration_error(target, probability),
        ace=adaptive_calibration_error(target, probability),
        sensitivity=sensitivity(target, prediction),
        specificity=specificity(target, prediction),
        positive_predictive_value=positive_predictive_value(target, prediction),
        negative_predictive_value=negative_predictive_value(target, prediction),
        calibration_intercept=intercept,
        calibration_slope=slope,
        hosmer_lemeshow_statistic=hl_statistic,
        hosmer_lemeshow_p_value=hl_p,
    )


@dataclass(frozen=True)
class SurvivalReport:
    concordance: float
    hazard_ratio: float
    hazard_lower: float
    hazard_upper: float
    hazard_p_value: float
    logrank_statistic: float
    logrank_p_value: float


def survival_report(
    time: FloatArray,
    event: IntArray,
    risk: FloatArray,
    group: IntArray,
) -> SurvivalReport:
    hazard, lower, upper, hazard_p = hazard_ratio_binary(time, event, group)
    logrank, logrank_p = logrank_test(time, event, group)
    return SurvivalReport(
        concordance=concordance_index(time, risk, event),
        hazard_ratio=hazard,
        hazard_lower=lower,
        hazard_upper=upper,
        hazard_p_value=hazard_p,
        logrank_statistic=logrank,
        logrank_p_value=logrank_p,
    )
