from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from glioma_shift_atlas.contracts import EncodedModalities, ModelConfig, PatientBatch, PatientOutput


class FeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.input = nn.Linear(dimension, hidden)
        self.output = nn.Linear(hidden, dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.input(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.value(x) * torch.sigmoid(self.gate(x))


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.missing = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, x: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if x is None:
            return self.missing.view(1, -1).expand(batch_size, -1).to(device)
        return self.layers(torch.nan_to_num(x))


class PatchStem(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        widths = (64, 128, 256, 512)
        blocks: list[nn.Module] = []
        input_channels = 3
        for width in widths:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, width, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                ]
            )
            input_channels = width
        self.features = nn.Sequential(*blocks)
        self.projection = nn.Linear(widths[-1], output_dim)

    def forward(self, patches: Tensor) -> Tensor:
        batch, count, channels, height, width = patches.shape
        flattened = patches.reshape(batch * count, channels, height, width)
        encoded = self.features(flattened)
        encoded = encoded.mean(dim=(-2, -1))
        encoded = self.projection(encoded)
        return encoded.reshape(batch, count, -1)


class CoordinateEncoding(nn.Module):
    def __init__(self, dimension: int, frequencies: int = 32) -> None:
        super().__init__()
        self.dimension = dimension
        self.frequencies = frequencies
        self.projection = nn.Sequential(
            nn.Linear(frequencies * 4, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        scale = torch.arange(self.frequencies, device=coordinates.device, dtype=coordinates.dtype)
        scale = 2.0 ** scale
        angles = coordinates.unsqueeze(-1) * scale.view(1, 1, 1, -1) * math.pi
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        encoded = encoded.flatten(start_dim=-2)
        return self.projection(encoded)


class ExpressionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_coordinates: bool) -> None:
        super().__init__()
        self.use_coordinates = use_coordinates
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = GatedLinearUnit(input_dim, output_dim)
        self.coordinate = CoordinateEncoding(output_dim) if use_coordinates else None
        self.output = nn.Sequential(
            FeedForward(output_dim, expansion=2),
            FeedForward(output_dim, expansion=2),
        )

    def forward(self, expression: Tensor, coordinates: Tensor | None = None) -> Tensor:
        x = torch.log1p(torch.clamp_min(expression, 0.0))
        library = x.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x / library * 1e4
        x = self.input_norm(x)
        x = self.input_projection(x)
        if self.use_coordinates:
            if coordinates is None or self.coordinate is None:
                raise ValueError("coordinates are required")
            x = x + self.coordinate(coordinates)
        return self.output(x)


class BulkEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            GatedLinearUnit(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, expression: Tensor) -> Tensor:
        normalized = torch.log1p(torch.clamp_min(expression, 0.0))
        return self.network(normalized)


class MultiheadSetBlock(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        key_padding_mask = None if mask is None else ~mask.bool()
        attended, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm(x + attended)
        return self.feed_forward(x)


class SeedPooling(nn.Module):
    def __init__(self, dimension: int, heads: int, seeds: int = 1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(seeds, dimension))
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension)
        nn.init.normal_(self.seeds, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch = x.shape[0]
        queries = self.seeds.unsqueeze(0).expand(batch, -1, -1)
        key_padding_mask = None if mask is None else ~mask.bool()
        pooled, _ = self.attention(
            queries,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm(queries + pooled)
        return self.feed_forward(pooled)


class NicheSetTransformer(nn.Module):
    def __init__(self, input_dim: int, patient_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, patient_dim)
        self.layers = nn.ModuleList(
            MultiheadSetBlock(patient_dim, heads) for _ in range(layers)
        )
        self.pooling = SeedPooling(patient_dim, heads)

    def forward(self, tokens: Tensor, mask: Tensor | None) -> Tensor:
        x = self.projection(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        return self.pooling(x, mask).squeeze(1)


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            FeedForward(output_dim, expansion=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class RoutingExpert(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
            nn.LayerNorm(dimension),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class FlexMixture(nn.Module):
    def __init__(self, dimension: int, modalities: int, experts: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.modalities = modalities
        self.experts = experts
        self.missing = nn.Parameter(torch.empty(modalities, dimension))
        self.availability = nn.Sequential(
            nn.Linear(modalities, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.router = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Linear(dimension, experts),
        )
        self.expert_bank = nn.ModuleList(RoutingExpert(dimension) for _ in range(experts))
        self.output = nn.LayerNorm(dimension)
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, tokens: Tensor, availability: Tensor) -> tuple[Tensor, Tensor]:
        missing = self.missing.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        observed = availability.unsqueeze(-1).bool()
        complete = torch.where(observed, tokens, missing)
        denominator = availability.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (complete * availability.unsqueeze(-1)).sum(dim=1) / denominator
        availability_embedding = self.availability(availability.float())
        routing_input = torch.cat((pooled, availability_embedding), dim=-1)
        weights = torch.softmax(self.router(routing_input), dim=-1)
        expert_outputs = torch.stack([expert(pooled) for expert in self.expert_bank], dim=1)
        fused = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        return self.output(fused + pooled), weights


class IDHConditionalAdapter(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.wildtype = RoutingExpert(dimension)
        self.mutant = RoutingExpert(dimension)
        self.inferred_gate = nn.Linear(dimension, 1)
        self.output = nn.LayerNorm(dimension)

    def forward(self, x: Tensor, status: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        inferred = torch.sigmoid(self.inferred_gate(x)).squeeze(-1)
        gate = torch.where(observed.bool(), status.float(), inferred)
        wildtype = self.wildtype(x)
        mutant = self.mutant(x)
        adapted = wildtype * (1.0 - gate.unsqueeze(-1)) + mutant * gate.unsqueeze(-1)
        return self.output(x + adapted), inferred


class VariationalPrototypeHead(nn.Module):
    def __init__(self, dimension: int, prototypes: int) -> None:
        super().__init__()
        self.mean = nn.Linear(dimension, dimension)
        self.log_variance = nn.Linear(dimension, dimension)
        self.prototypes = nn.Parameter(torch.empty(prototypes, dimension))
        self.prior_logits = nn.Parameter(torch.zeros(prototypes))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.prototypes, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean = self.mean(x)
        log_variance = self.log_variance(x).clamp(-12.0, 8.0)
        if self.training:
            noise = torch.randn_like(mean)
            latent = mean + noise * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        latent = F.normalize(latent, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        temperature = self.temperature.abs().clamp_min(0.05)
        logits = latent @ prototypes.t() / temperature
        return logits, mean, log_variance, latent

    def categorical_kl(self, probabilities: Tensor) -> Tensor:
        prior = torch.softmax(self.prior_logits, dim=-1)
        value = probabilities * (
            torch.log(probabilities.clamp_min(1e-8)) - torch.log(prior.clamp_min(1e-8))
        )
        return value.sum(dim=-1).mean()


class PredictionHeads(nn.Module):
    def __init__(self, dimension: int, shift_dim: int) -> None:
        super().__init__()
        self.shift = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, shift_dim),
        )
        self.risk = nn.Linear(dimension, 1)
        self.therapy = nn.Linear(dimension, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.shift(x), self.risk(x).squeeze(-1), self.therapy(x).squeeze(-1)


class GliomaShiftAtlas(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        spatial_genes: int,
        single_cell_genes: int,
        bulk_genes: int,
        shift_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.histology = PatchStem(config.histology_output)
        self.spatial = ExpressionEncoder(spatial_genes, config.spatial_output, True)
        self.single_cell = ExpressionEncoder(single_cell_genes, config.transcript_output, False)
        self.bulk = BulkEncoder(bulk_genes, config.transcript_output)
        self.clinical = ClinicalEncoder(
            config.clinical_dim,
            config.clinical_hidden,
            config.clinical_output,
        )
        self.histology_set = NicheSetTransformer(
            config.histology_output,
            config.patient_dim,
            config.set_layers,
            config.set_heads,
        )
        self.spatial_set = NicheSetTransformer(
            config.spatial_output,
            config.patient_dim,
            config.set_layers,
            config.set_heads,
        )
        self.single_cell_set = NicheSetTransformer(
            config.transcript_output,
            config.patient_dim,
            config.set_layers,
            config.set_heads,
        )
        self.bulk_projection = ModalityProjector(config.transcript_output, config.patient_dim)
        self.clinical_projection = ModalityProjector(config.clinical_output, config.patient_dim)
        self.fusion = FlexMixture(config.patient_dim, 5, config.experts)
        self.idh_adapter = IDHConditionalAdapter(config.patient_dim)
        self.prototype = VariationalPrototypeHead(config.patient_dim, config.prototypes)
        self.heads = PredictionHeads(config.patient_dim, shift_dim)

    def _zeros(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.config.patient_dim, device=device)

    def encode_modalities(self, batch: PatientBatch) -> EncodedModalities:
        device = batch.modality_mask.device
        size = batch.size
        outputs: list[Tensor] = []
        if batch.histology is None:
            outputs.append(self._zeros(size, device))
        else:
            histology = self.histology(batch.histology)
            outputs.append(self.histology_set(histology, batch.histology_mask))
        if batch.spatial_expression is None:
            outputs.append(self._zeros(size, device))
        else:
            spatial = self.spatial(batch.spatial_expression, batch.spatial_coordinates)
            outputs.append(self.spatial_set(spatial, batch.spatial_mask))
        if batch.single_cell_expression is None:
            outputs.append(self._zeros(size, device))
        else:
            cells = self.single_cell(batch.single_cell_expression)
            outputs.append(self.single_cell_set(cells, batch.single_cell_mask))
        if batch.bulk_expression is None:
            outputs.append(self._zeros(size, device))
        else:
            outputs.append(self.bulk_projection(self.bulk(batch.bulk_expression)))
        clinical = self.clinical(batch.clinical, size, device)
        outputs.append(self.clinical_projection(clinical))
        return EncodedModalities(
            tokens=torch.stack(outputs, dim=1),
            availability=batch.modality_mask,
            modality_names=("histology", "spatial", "single_cell", "bulk", "clinical"),
        )

    def forward(self, batch: PatientBatch, availability: Tensor | None = None) -> PatientOutput:
        modalities = self.encode_modalities(batch)
        active = modalities.availability if availability is None else availability
        fused, routing = self.fusion(modalities.tokens, active)
        adapted, idh_logit = self.idh_adapter(
            fused,
            batch.idh_status,
            batch.idh_observed,
        )
        logits, mean, log_variance, embedding = self.prototype(adapted)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, subtype = probabilities.max(dim=-1)
        shift, risk, therapy = self.heads(embedding)
        return PatientOutput(
            embedding=embedding,
            fused_embedding=fused,
            idh_embedding=adapted,
            prototype_logits=logits,
            prototype_probabilities=probabilities,
            subtype=subtype,
            confidence=confidence,
            shift=shift,
            risk=risk,
            therapy_logit=therapy,
            idh_logit=idh_logit,
            variational_mean=mean,
            variational_log_variance=log_variance,
            routing_weights=routing,
        )

    def parameter_groups(self, backbone_learning_rate: float, head_learning_rate: float) -> list[dict[str, object]]:
        backbone_modules: Sequence[nn.Module] = (
            self.histology,
            self.spatial,
            self.single_cell,
            self.bulk,
        )
        backbone_parameters = [
            parameter
            for module in backbone_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        return [
            {"params": backbone_parameters, "lr": backbone_learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ]

    def freeze_encoders(self) -> None:
        for module in (self.histology, self.spatial, self.single_cell, self.bulk):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def unfreeze_encoders(self) -> None:
        for module in (self.histology, self.spatial, self.single_cell, self.bulk):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
