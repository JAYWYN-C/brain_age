# brain_age

## Brain-Age Pretraining for EEG Dementia Classification

This repository provides the source code accompanying our manuscript on brain-age pretraining for subject-independent, four-class EEG dementia classification.

The downstream classification task includes the following diagnostic groups:

* Cognitively normal controls (CN)
* Mild cognitive impairment (MCI)
* Alzheimer's disease (AD)
* Non-Alzheimer's dementia

The code is released to support methodological transparency, reproducibility, and independent audit. Raw data, derived dataset arrays, subject annotations, trained model checkpoints, experimental results, generated figures, and local configuration files are not included.

> **Disclaimer:** This repository is intended for research use only. It is not a clinical decision-support or diagnostic system.

## Repository Structure

```text
.
├── main_model/
├── experiments/
│   ├── baselines/
│   └── baseline_models/
└── README.md
```

### `main_model/`

Contains the final implementation of the proposed model used in the manuscript:

* `BasicDeepCNN`
* CN-only brain-age pretraining
* Downstream four-class diagnosis training
* Subject-independent data split handling
* Data transformation utilities
* Bundled public configuration defaults
* Runtime and training utilities

### `experiments/baselines/`

Contains orchestration utilities for the retained V0 and V1 baseline experiments.

### `experiments/baseline_models/`

Contains the baseline model implementations retained for the manuscript comparison:

* ResNet1D
* ShallowFBCSPNet
* VGG1D
* MSNN

Only Python source files and this README are intended to be tracked in the public repository.

## Data Availability and Exclusions

The CAUEEG-derived data are intentionally excluded from this repository.

These files must be obtained, stored, and processed separately in accordance with the applicable dataset license, institutional approval, and data-use conditions.

## Final EEG Preprocessing Configuration

The manuscript uses the following final preprocessing configuration:

| Parameter         | Setting                    |
| ----------------- | -------------------------- |
| EEG channels      | 19 scalp channels          |
| Electrode system  | International 10–20 system |
| Reference         | Common average reference   |
| Band-pass filter  | 0.5–45 Hz                  |
| Epoch duration    | 4 seconds                  |
| Epoch overlap     | 0 seconds                  |
| Sampling rate     | 200 Hz                     |
| Samples per epoch | 800                        |

The public `main_model/dataset_catalog.py` retains only the final dataset alias corresponding to the 4-second, non-overlapping, 0.5–45 Hz preprocessing configuration.

## Experimental Variants

Two experimental variants are compared.

### V0: Vanilla

The encoder is randomly initialized and trained directly on the downstream four-class diagnosis task.

### V1: Brain-Age Pretrained

The encoder is first pretrained to predict chronological age using EEG data from cognitively normal subjects only. The pretrained encoder is then used to initialize the downstream four-class diagnosis model and is fine-tuned on the same diagnosis task as V0.

V0 and V1 use identical:

* Subject-independent data folds
* EEG preprocessing
* Downstream optimization settings
* Training epoch budgets
* Validation-based model-selection procedures

Therefore, the intended experimental difference between V0 and V1 is the encoder initialization strategy.

## Evaluation Protocol

All downstream evaluation metrics are computed at the **subject level**, not at the individual epoch level.

For each subject:

1. Obtain the model logits for all EEG epochs belonging to that subject.
2. Average the epoch-level logits.
3. Apply softmax to the averaged logits to obtain subject-level class probabilities.
4. Compute predictions and evaluation metrics from the resulting subject-level outputs.

This aggregation procedure prevents subjects with larger numbers of epochs from contributing disproportionately to the reported performance.


## Retained Model Suite

The final model suite included in the public repository is:

| Model           | Role           |
| --------------- | -------------- |
| BasicDeepCNN    | Proposed model |
| ResNet1D        | Baseline       |
| ShallowFBCSPNet | Baseline       |
| VGG1D           | Baseline       |
| MSNN            | Baseline       |

Earlier development workspaces included helper paths for TinyCNN1D and Deep4Net. These models were not retained in the final manuscript comparison and are therefore not included in this public release.
