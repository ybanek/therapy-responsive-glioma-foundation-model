# Therapy-Responsive Glioma Microenvironment Profiling

This repository contains the training and evaluation system for a multimodal glioma model that combines histology, spatial transcriptomics, single-cell transcriptomics, bulk molecular profiles, and clinical variables. A prototype information bottleneck maps patient representations to six microenvironment subtypes, while a predicted post-therapy compositional-shift objective, missing-modality fusion, cross-cohort alignment, and an IDH-conditional adapter shape the shared representation.

## Installation

Python 3.11, CUDA 12.1, and an eight-GPU NVIDIA A100 80 GB node are the reference environment.

```bash
conda env create -f environment.yml
conda activate glioma-shift-atlas
pip install -e .
```

The container image can be built with `docker build -t glioma-shift-atlas .`.

## Data

Verified source endpoints are listed in `dataset_links.txt`. TCGA controlled clinical and genomic fields require GDC or dbGaP authorization. CPTAC and CGGA access remains subject to their portal terms. Prepare a manifest with patient identifiers replaced by local study keys, modality paths, cohort labels, outcome time, event state, therapy response, and molecular covariates.

```bash
glioma-atlas prepare --manifest manifests/raw.csv --destination prepared
```

The prepared manifest is content-addressed. Record its SHA-256 and expected storage footprint in the local run registry before training. Whole-slide images and spatial matrices can require several terabytes depending on selected cases and magnification.

## Training

The primary configuration preserves the reported two-stage schedule: 30 pretraining epochs followed by 50 fine-tuning epochs, batch size 32, AdamW at `1e-4`, cosine decay, weight decay `1e-4`, gradient clipping at 1.0, and 15 seeds. Pretraining ramps modality dropout from 0 to 0.3 after a ten-epoch warm-up.

```bash
torchrun --nproc_per_node=8 -m glioma_shift_atlas.console train --config settings/main.yaml
```

A complete primary run uses 8 NVIDIA A100 80 GB GPUs and 1 TB host memory. The reported program consumed about 5,500 GPU-hours for pretraining and about 500 GPU-hours per cohort for fine-tuning; the full seed and ablation program totaled about 22,000 GPU-hours.

## Evaluation

```bash
torchrun --nproc_per_node=8 -m glioma_shift_atlas.console evaluate --config settings/main.yaml --cohort TCGA-GBM
glioma-atlas summarize --runs outputs --destination reports
```

The primary reference values are TCGA-GBM survival C-index 0.852 with bootstrap interval 0.832–0.871, TCGA-LGG C-index 0.871 with interval 0.852–0.889, CGGA C-index 0.841 with interval 0.820–0.861, CPTAC-GBM C-index 0.806 with interval 0.781–0.829, and TCGA-GBM therapy-response AUC 0.831. Fifteen seeds and five-fold cross-validation are required for comparison with these values. Dataset revisions, access-controlled fields, pretrained encoder weights, and hardware kernels can affect measured results.

## Commands

`glioma-atlas inspect` validates manifests and modality shapes. `glioma-atlas pretrain` runs the shift and alignment stage. `glioma-atlas finetune` runs cohort-specific survival, response, and subtype optimization. `glioma-atlas evaluate` computes discrimination, calibration, survival, reclassification, subgroup, and heterogeneity statistics. `glioma-atlas infer` emits subtype, confidence, compositional shift, survival risk, and therapy-response probability.

## License

The software is distributed under the BSD 3-Clause license. Dataset licenses and data-use agreements remain independent.
