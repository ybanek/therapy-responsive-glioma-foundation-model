from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CohortDefinition:
    name: str
    country: str
    patients: int
    modalities: tuple[str, ...]
    primary_endpoint: str
    external: bool
    spatial_resolution_microns: float | None


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    cohort: str
    stage: str
    objective: str
    metric: str
    removed_components: tuple[str, ...]
    modality_subset: tuple[str, ...]
    prototype_count: int
    ib_beta: float
    shift_weight: float
    clinical_weight: float
    alignment_weight: float
    modality_weight: float


ANCHOR_COHORTS = (
    CohortDefinition(
        name="TCGA-GBM",
        country="USA",
        patients=612,
        modalities=("histology", "bulk", "methylation", "copy_number", "clinical"),
        primary_endpoint="overall_survival",
        external=False,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="TCGA-LGG",
        country="USA",
        patients=516,
        modalities=("histology", "bulk", "methylation", "copy_number", "clinical"),
        primary_endpoint="overall_survival",
        external=False,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="Visium-GBM",
        country="International",
        patients=28,
        modalities=("histology", "spatial", "clinical"),
        primary_endpoint="niche_macro_f1",
        external=False,
        spatial_resolution_microns=50.0,
    ),
    CohortDefinition(
        name="CPTAC-GBM",
        country="USA",
        patients=99,
        modalities=("histology", "bulk", "proteomics", "phosphoproteomics", "clinical"),
        primary_endpoint="overall_survival",
        external=False,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="CGGA",
        country="China",
        patients=2000,
        modalities=("bulk", "methylation", "clinical"),
        primary_endpoint="overall_survival",
        external=True,
        spatial_resolution_microns=None,
    ),
)

EXTENDED_COHORTS = (
    CohortDefinition(
        name="IvyGAP",
        country="USA",
        patients=270,
        modalities=("anatomic_bulk", "histology", "clinical"),
        primary_endpoint="anatomic_transfer",
        external=True,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="GLASS",
        country="International",
        patients=200,
        modalities=("bulk", "methylation", "clinical", "longitudinal"),
        primary_endpoint="compositional_shift",
        external=True,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="GBmap",
        country="International",
        patients=0,
        modalities=("single_cell", "spatial"),
        primary_endpoint="cell_state_transfer",
        external=True,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="Neftel",
        country="USA",
        patients=0,
        modalities=("single_cell",),
        primary_endpoint="cell_state_transfer",
        external=True,
        spatial_resolution_microns=None,
    ),
    CohortDefinition(
        name="Couturier",
        country="France",
        patients=0,
        modalities=("single_cell",),
        primary_endpoint="cell_state_transfer",
        external=True,
        spatial_resolution_microns=None,
    ),
)

FULL_COMPONENTS = (
    "prototype_information_bottleneck",
    "predicted_compositional_shift",
    "flex_mixture_fusion",
    "niche_set_transformer",
    "idh_conditional_adapter",
    "optimal_transport_alignment",
    "modality_dropout_consistency",
)

MODALITIES = (
    "histology",
    "spatial",
    "single_cell",
    "bulk",
    "clinical",
)

REFERENCE_SEEDS = (
    42,
    1024,
    2048,
    3407,
    7,
    13,
    73,
    1729,
    31415,
    271828,
    161803,
    6022,
    8675,
    9000,
    4242,
)

PRIMARY_EXPERIMENT = ExperimentDefinition(
    name="primary",
    cohort="TCGA-GBM",
    stage="finetune",
    objective="joint",
    metric="c_index",
    removed_components=(),
    modality_subset=MODALITIES,
    prototype_count=6,
    ib_beta=0.1,
    shift_weight=1.0,
    clinical_weight=0.5,
    alignment_weight=0.2,
    modality_weight=0.3,
)


def primary_experiments() -> tuple[ExperimentDefinition, ...]:
    output: list[ExperimentDefinition] = []
    for cohort in ANCHOR_COHORTS:
        metric = "niche_macro_f1" if cohort.name == "Visium-GBM" else "c_index"
        output.append(
            replace(
                PRIMARY_EXPERIMENT,
                name=f"primary_{cohort.name.lower()}",
                cohort=cohort.name,
                metric=metric,
            )
        )
    return tuple(output)


def component_ablations() -> tuple[ExperimentDefinition, ...]:
    output: list[ExperimentDefinition] = []
    for component in FULL_COMPONENTS:
        metric = "c_index"
        if component in ("prototype_information_bottleneck", "niche_set_transformer"):
            metric = "subtype_kappa"
        if component in ("predicted_compositional_shift", "modality_dropout_consistency"):
            metric = "therapy_auc"
        if component == "optimal_transport_alignment":
            output.append(
                replace(
                    PRIMARY_EXPERIMENT,
                    name=f"remove_{component}_cgga",
                    cohort="CGGA",
                    metric=metric,
                    removed_components=(component,),
                )
            )
        output.append(
            replace(
                PRIMARY_EXPERIMENT,
                name=f"remove_{component}",
                metric=metric,
                removed_components=(component,),
            )
        )
    return tuple(output)


def prototype_sweep() -> tuple[ExperimentDefinition, ...]:
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=f"prototype_count_{count}",
            prototype_count=count,
            metric="therapy_auc",
        )
        for count in (2, 4, 6, 8, 10)
    )


def beta_sweep() -> tuple[ExperimentDefinition, ...]:
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=f"ib_beta_{str(value).replace('.', '_')}",
            ib_beta=value,
            metric="therapy_auc",
        )
        for value in (0.01, 0.05, 0.1, 0.5, 1.0)
    )


def shift_weight_sweep() -> tuple[ExperimentDefinition, ...]:
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=f"shift_weight_{str(value).replace('.', '_')}",
            shift_weight=value,
            metric="therapy_auc",
        )
        for value in (0.5, 0.75, 1.0, 1.5, 2.0)
    )


def modality_ablations() -> tuple[ExperimentDefinition, ...]:
    output: list[ExperimentDefinition] = []
    for modality in MODALITIES:
        subset = tuple(item for item in MODALITIES if item != modality)
        output.append(
            replace(
                PRIMARY_EXPERIMENT,
                name=f"without_{modality}",
                modality_subset=subset,
            )
        )
    for modality in MODALITIES:
        output.append(
            replace(
                PRIMARY_EXPERIMENT,
                name=f"only_{modality}",
                modality_subset=(modality,),
            )
        )
    return tuple(output)


def negative_controls() -> tuple[ExperimentDefinition, ...]:
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=name,
            objective=name,
            metric="hazard_ratio",
        )
        for name in (
            "shuffled_subtype_labels",
            "random_prototype_assignment",
            "scrambled_shift_labels",
            "random_feature_permutation",
            "permutation_distribution",
        )
    )


def robustness_experiments() -> tuple[ExperimentDefinition, ...]:
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=name,
            objective=name,
            metric="c_index",
        )
        for name in (
            "macenco_stain_normalization",
            "jpeg_quality_80",
            "jpeg_quality_50",
            "gaussian_noise_0_01",
            "gaussian_noise_0_05",
            "pgd_2_255",
        )
    )


def subgroup_experiments() -> tuple[ExperimentDefinition, ...]:
    levels = (
        "idh_mutant",
        "idh_wildtype",
        "mgmt_methylated",
        "mgmt_unmethylated",
        "age_below_50",
        "age_50_to_65",
        "age_above_65",
        "female",
        "male",
        "grade_2",
        "grade_3",
        "grade_4",
        "primary",
        "recurrent",
    )
    return tuple(
        replace(
            PRIMARY_EXPERIMENT,
            name=f"subgroup_{level}",
            objective=level,
            metric="therapy_auc",
        )
        for level in levels
    )


def all_experiments() -> tuple[ExperimentDefinition, ...]:
    return (
        *primary_experiments(),
        *component_ablations(),
        *prototype_sweep(),
        *beta_sweep(),
        *shift_weight_sweep(),
        *modality_ablations(),
        *negative_controls(),
        *robustness_experiments(),
        *subgroup_experiments(),
    )


def experiment_by_name(name: str) -> ExperimentDefinition:
    matches = [experiment for experiment in all_experiments() if experiment.name == name]
    if len(matches) != 1:
        raise KeyError(name)
    return matches[0]


def experiments_for_cohort(cohort: str) -> tuple[ExperimentDefinition, ...]:
    return tuple(experiment for experiment in all_experiments() if experiment.cohort == cohort)


def experiments_by_metric(metric: str) -> tuple[ExperimentDefinition, ...]:
    return tuple(experiment for experiment in all_experiments() if experiment.metric == metric)


def iter_seeded_experiments(
    seeds: Sequence[int],
) -> Iterator[tuple[ExperimentDefinition, int]]:
    for experiment in all_experiments():
        for seed in seeds:
            yield experiment, seed


def validate_registry() -> None:
    experiments = all_experiments()
    names = [experiment.name for experiment in experiments]
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be unique")
    cohorts = {cohort.name for cohort in (*ANCHOR_COHORTS, *EXTENDED_COHORTS)}
    for experiment in experiments:
        if experiment.cohort not in cohorts:
            raise ValueError(f"unknown cohort: {experiment.cohort}")
        if experiment.prototype_count < 2:
            raise ValueError("prototype count must be at least two")
        if experiment.ib_beta <= 0:
            raise ValueError("IB beta must be positive")
        if not set(experiment.modality_subset).issubset(MODALITIES):
            raise ValueError("unknown modality")
