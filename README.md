# dissociating-aptitude-multilingual-experience_Balboni_et_al_2026


## 📝 Project Overview & Abstract

This repository contains the analysis pipeline, behavioural datasets, and functional MRI (fMRI) region-of-interest extractions and design files for investigating how individual differences in **language learning aptitude** and **multilingual speech experience** interactively modulate brain activity during speech perception.

### Methodology Summary
* ** Behavioural Profiling:** Multilingual language profiles and speaking entropy indices were collected via the LEAP-Q (`participant_lang_profile.tsv`, `leapq_entropy_competence_filtered.tsv`). Individual aptitude subtest scores were normalised into composite Z-scores (`participants_aptitude_individual_scores_imputed.tsv`).
* **fMRI Analysis:** Higher-level group analyses were conducted using FSL FEAT (thresholded at $Z > 2.3$, $p < 0.05$ cluster-corrected). Design files and resulting significance clusters are available within cope1 directories 

---

## 📁 Repository Structure

```text
dissociating-aptitude-multilingual-experience_Balboni_et_al_2026/
├── 01_behavioral_correlations.py
├── 02_plot_individual_cluster_activity_vs_beh.py
├── behavioural_data/
│   ├── leapq_entropy_competence_filtered.tsv
│   ├── participant_lang_profile.tsv
│   └── participants_aptitude_individual_scores_imputed.tsv
├── fsl/
│   ├── degraded.vs.baseline.thr2.3.covariates.gfeat/
│   │   ├── cope1.feat/
│   │   │   └── thresh_zstat*.nii.gz
│   │   ├── degraded_vs_baseline_contrast_2_cluster_activity.csv
│   │   ├── degraded_vs_baseline_contrast_3_cluster_activity.csv
│   │   ├── degraded_vs_baseline_contrast_4_cluster_activity.csv
│   │   ├── degraded_vs_baseline_contrast_6_cluster_activity.csv
│   │   ├── degraded_vs_baseline_contrast_7_cluster_activity.csv
│   │   ├── design.fsf
│   │   └── summary_degraded.vs.baseline.thr2.3.covariates.tsv
│   ├── intact.vs.baseline.thr2.3.covariates.gfeat/
│   │   ├── cope1.feat/
│   │   │   └── thresh_zstat*.nii.gz
│   │   ├── design.fsf
│   │   ├── intact_vs_baseline_contrast_2_cluster_activity.csv
│   │   ├── intact_vs_baseline_contrast_5_cluster_activity.csv
│   │   ├── intact_vs_baseline_contrast_6_cluster_activity.csv
│   │   └── summary_intact.vs.baseline.thr2.3.covariates.tsv
│   └── intact.vs.degraded.thr2.3.covariates.gfeat/
│       ├── cope1.feat/
│       │   └── thresh_zstat*.nii.gz
│       ├── design.fsf
│       ├── intact_vs_degraded_contrast_2_cluster_activity.csv
│       ├── intact_vs_degraded_contrast_5_cluster_activity.csv
│       └── summary_intact.vs.degraded.thr2.3.covariates.tsv
└── output/
