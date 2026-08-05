from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from glioma_shift_atlas.contracts import LossBundle, LossConfig, PatientBatch, PatientOutput


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    value = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp())
    return value.sum(dim=-1).mean()


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    normalized = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(normalized * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def cox_partial_likelihood(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order].to(risk.dtype)
    log_denominator = torch.logcumsumexp(ordered_risk, dim=0)
    contributions = ordered_risk - log_denominator
    return -(contributions * ordered_event).sum() / ordered_event.sum().clamp_min(1.0)


def therapy_binary_cross_entropy(logits: Tensor, targets: Tensor, observed: Tensor) -> Tensor:
    values = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return masked_mean(values, observed)


def shift_mean_squared_error(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    values = (prediction - target).square().mean(dim=-1)
    return masked_mean(values, observed)


def modality_consistency(reference: Tensor, perturbed: Tensor) -> Tensor:
    reference = F.normalize(reference, dim=-1)
    perturbed = F.normalize(perturbed, dim=-1)
    return (reference - perturbed).square().sum(dim=-1).mean()


def pairwise_squared_distance(x: Tensor, y: Tensor) -> Tensor:
    x_norm = x.square().sum(dim=-1, keepdim=True)
    y_norm = y.square().sum(dim=-1, keepdim=True).transpose(0, 1)
    return (x_norm + y_norm - 2.0 * x @ y.transpose(0, 1)).clamp_min(0.0)


def sinkhorn_transport_cost(
    source: Tensor,
    target: Tensor,
    epsilon: float = 0.05,
    iterations: int = 50,
) -> Tensor:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return source.new_zeros(())
    cost = pairwise_squared_distance(source, target)
    log_kernel = -cost / epsilon
    log_a = source.new_full((source.shape[0],), -math.log(source.shape[0]))
    log_b = target.new_full((target.shape[0],), -math.log(target.shape[0]))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    transport = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    return (transport * cost).sum()


def multicohort_alignment(embedding: Tensor, cohorts: list[str]) -> Tensor:
    unique = sorted(set(cohorts))
    if len(unique) < 2:
        return embedding.new_zeros(())
    costs: list[Tensor] = []
    for left_index, left in enumerate(unique):
        left_mask = torch.tensor([item == left for item in cohorts], device=embedding.device)
        left_values = embedding[left_mask]
        for right in unique[left_index + 1 :]:
            right_mask = torch.tensor([item == right for item in cohorts], device=embedding.device)
            right_values = embedding[right_mask]
            costs.append(sinkhorn_transport_cost(left_values, right_values))
    return torch.stack(costs).mean() if costs else embedding.new_zeros(())


def balanced_routing_loss(weights: Tensor) -> Tensor:
    experts = weights.shape[-1]
    mean_probability = weights.mean(dim=0)
    target = torch.full_like(mean_probability, 1.0 / experts)
    return F.mse_loss(mean_probability, target)


def prototype_separation_loss(prototypes: Tensor, margin: float = 0.2) -> Tensor:
    normalized = F.normalize(prototypes, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    identity = torch.eye(similarities.shape[0], device=similarities.device, dtype=torch.bool)
    off_diagonal = similarities.masked_select(~identity)
    return F.relu(off_diagonal - margin).mean()


def prototype_occupancy_loss(probabilities: Tensor) -> Tensor:
    occupancy = probabilities.mean(dim=0)
    uniform = torch.full_like(occupancy, 1.0 / occupancy.numel())
    return F.kl_div(occupancy.clamp_min(1e-8).log(), uniform, reduction="sum")


def entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)


def confidence_regularizer(probabilities: Tensor, target_entropy: float = 0.5) -> Tensor:
    values = entropy(probabilities)
    return (values - target_entropy).square().mean()


def idh_binary_loss(logits: Tensor, status: Tensor, observed: Tensor) -> Tensor:
    values = F.binary_cross_entropy(logits.clamp(1e-6, 1.0 - 1e-6), status.float(), reduction="none")
    return masked_mean(values, observed)


def symmetric_kl(first: Tensor, second: Tensor) -> Tensor:
    first_log = F.log_softmax(first, dim=-1)
    second_log = F.log_softmax(second, dim=-1)
    first_probability = first_log.exp()
    second_probability = second_log.exp()
    left = F.kl_div(first_log, second_probability, reduction="batchmean")
    right = F.kl_div(second_log, first_probability, reduction="batchmean")
    return 0.5 * (left + right)


def pseudo_shift_targets(
    observed_shift: Tensor,
    observed: Tensor,
    subtype_scores: Tensor,
    therapy: Tensor,
    mgmt: Tensor,
    idh: Tensor,
) -> tuple[Tensor, Tensor]:
    target = observed_shift.clone()
    weights = observed.float().clone()
    missing_indices = torch.nonzero(~observed.bool(), as_tuple=False).flatten()
    observed_indices = torch.nonzero(observed.bool(), as_tuple=False).flatten()
    if missing_indices.numel() == 0 or observed_indices.numel() == 0:
        return target, weights
    covariates = torch.cat(
        (
            subtype_scores,
            therapy.float().unsqueeze(-1),
            mgmt.float().unsqueeze(-1),
            idh.float().unsqueeze(-1),
        ),
        dim=-1,
    )
    observed_covariates = covariates[observed_indices]
    missing_covariates = covariates[missing_indices]
    distances = torch.cdist(missing_covariates, observed_covariates)
    neighbors = distances.topk(min(8, observed_indices.numel()), largest=False, dim=-1)
    neighbor_weights = torch.softmax(-neighbors.values, dim=-1)
    neighbor_targets = observed_shift[observed_indices[neighbors.indices]]
    estimates = (neighbor_targets * neighbor_weights.unsqueeze(-1)).sum(dim=1)
    target[missing_indices] = estimates
    weights[missing_indices] = 0.5
    return target, weights


@dataclass(frozen=True)
class ObjectiveWeights:
    shift: float
    clinical: float
    alignment: float
    modality: float
    beta: float
    routing: float = 0.01
    occupancy: float = 0.01
    idh: float = 0.05

    @classmethod
    def from_config(cls, config: LossConfig, beta: float) -> ObjectiveWeights:
        return cls(
            shift=config.shift,
            clinical=config.clinical,
            alignment=config.alignment,
            modality=config.modality,
            beta=beta,
        )


class PretrainingObjective(nn.Module):
    def __init__(self, weights: ObjectiveWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        output: PatientOutput,
        reference: PatientOutput,
        batch: PatientBatch,
        shift_target: Tensor,
        shift_weight: Tensor,
    ) -> LossBundle:
        shift_values = (output.shift - shift_target).square().mean(dim=-1)
        shift = (shift_values * shift_weight).sum() / shift_weight.sum().clamp_min(1.0)
        alignment = multicohort_alignment(output.embedding, batch.cohorts)
        modality = modality_consistency(reference.embedding.detach(), output.embedding)
        routing = balanced_routing_loss(output.routing_weights)
        total = (
            self.weights.shift * shift
            + self.weights.alignment * alignment
            + self.weights.modality * modality
            + self.weights.routing * routing
        )
        zero = total.new_zeros(())
        return LossBundle(
            total=total,
            subtype=zero,
            shift=shift,
            cox=zero,
            therapy=zero,
            alignment=alignment,
            modality=modality,
            kl=zero,
            values={
                "total": float(total.detach()),
                "shift": float(shift.detach()),
                "alignment": float(alignment.detach()),
                "modality": float(modality.detach()),
                "routing": float(routing.detach()),
            },
        )


class FinetuningObjective(nn.Module):
    def __init__(self, weights: ObjectiveWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, output: PatientOutput, batch: PatientBatch) -> LossBundle:
        subtype = soft_cross_entropy(output.prototype_logits, batch.subtype_scores)
        shift = shift_mean_squared_error(output.shift, batch.compositional_shift, batch.shift_observed)
        cox = cox_partial_likelihood(output.risk, batch.survival_time, batch.survival_event)
        therapy = therapy_binary_cross_entropy(
            output.therapy_logit,
            batch.therapy_response,
            batch.therapy_observed,
        )
        kl = gaussian_kl(output.variational_mean, output.variational_log_variance)
        categorical = prototype_occupancy_loss(output.prototype_probabilities)
        routing = balanced_routing_loss(output.routing_weights)
        idh = idh_binary_loss(output.idh_logit, batch.idh_status, batch.idh_observed)
        clinical = cox + therapy
        total = (
            subtype
            + self.weights.beta * kl
            + self.weights.shift * shift
            + self.weights.clinical * clinical
            + self.weights.occupancy * categorical
            + self.weights.routing * routing
            + self.weights.idh * idh
        )
        zero = total.new_zeros(())
        return LossBundle(
            total=total,
            subtype=subtype,
            shift=shift,
            cox=cox,
            therapy=therapy,
            alignment=zero,
            modality=zero,
            kl=kl,
            values={
                "total": float(total.detach()),
                "subtype": float(subtype.detach()),
                "shift": float(shift.detach()),
                "cox": float(cox.detach()),
                "therapy": float(therapy.detach()),
                "kl": float(kl.detach()),
                "occupancy": float(categorical.detach()),
                "routing": float(routing.detach()),
                "idh": float(idh.detach()),
            },
        )


def modality_dropout_probability(epoch: int, warmup: int, ramp_end: int, final: float) -> float:
    if epoch <= warmup:
        return 0.0
    progress = min(1.0, max(0.0, (epoch - warmup) / (ramp_end - warmup)))
    return final * progress


def sample_modality_mask(availability: Tensor, probability: float) -> Tensor:
    if probability <= 0.0:
        return availability
    keep = torch.rand_like(availability.float()) >= probability
    sampled = availability.bool() & keep
    none_present = sampled.sum(dim=-1) == 0
    if none_present.any():
        rows = torch.nonzero(none_present, as_tuple=False).flatten()
        for row in rows.tolist():
            candidates = torch.nonzero(availability[row].bool(), as_tuple=False).flatten()
            if candidates.numel() > 0:
                selected = candidates[torch.randint(candidates.numel(), (1,), device=candidates.device)]
                sampled[row, selected] = True
    return sampled.to(availability.dtype)
