# Cross-dataset generalization of deep learning models for diabetic retinopathy detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23681312.svg)](https://doi.org/10.5281/zenodo.23681312)

Code, per-image predictions, and result tables for the article
*"Evaluating Cross-Dataset Generalization of Deep Learning Models for Diabetic Retinopathy Detection Using Retinal Fundus Images"*
by Isman Kurniawan, Prazna Paramitha Avi, Ghina Khoerunnisa, Harry Setiawan, Nor Aida Zuraimi Md Noar, and Mohana Sundaram Muthuvalu.

Four ImageNet-pretrained backbones (ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16) were trained on APTOS 2019
and evaluated without adaptation on IDRiD and Messidor. This record allows every table and figure of the article to be
reproduced from the released predictions and the official dataset labels.

## Contents

| Folder | Content |
|---|---|
| `predictions/<model>/pred_{val,idrid,messidor}.csv` | Predicted probability of DR (`y_prob`) per image (`id_code`) for 26 trained models on the APTOS validation split (733 images), IDRiD (455 images), and Messidor (1,200 images). Labels are not included (see below). |
| `results/T2.csv` ... `T7.csv` | Result tables of the article (main results, preprocessing ablation, decomposition and calibration, per-grade detection rate, label definitions, pretraining control). |
| `code/analysis/` | `merge_labels.py` (attach official labels), `reproduce_results.py` (recompute all tables), `make_tables.py` and `make_figures.py` (LaTeX tables and figures). |
| `code/inference_training/` | Inference (`predict_all.py`), retraining of the ResNet50 preprocessing series (`train_r50_prep.py`, `train_r50_5a_aug.py`), Graham normalization of the CLAHE configurations (`make_graham_6x.py`), Grad-CAM (`gradcam_f7.py`), and the model manifest. |
| `code/notebooks/` | Original preprocessing notebooks (circular crop, bounding-box crop, CLAHE, Graham normalization) and training notebooks of the models, with outputs removed. |

## Model identifiers

`<backbone>-<init>-<preprocessing>`, where backbone is `R50` (ResNet50), `D121` (DenseNet121), `EB0` (EfficientNet-B0), or `VIT` (ViT-B/16),
init is `P` (ImageNet-pretrained) or `NP` (trained from scratch), and preprocessing is `RAW` (none), `1A` (circular crop), `1B` (bounding-box crop),
`2A` (CLAHE), `3A` (Graham), `4A` (circular crop + CLAHE), `4B` (box crop + CLAHE), `5A` (circular crop + Graham), `5B` (box crop + Graham),
`6A` (circular crop + CLAHE + Graham), `6B` (box crop + CLAHE + Graham), or `5A-AUG` (5A with data augmentation, exploratory).
`R50-P-1A-ORIG` is a second, independent training run of `R50-P-1A` used to estimate the run-to-run variability.

## Datasets and labels

The dataset annotations are not redistributed here because of the dataset licenses. Please obtain the datasets from their official sources:

- APTOS 2019 Blindness Detection, https://www.kaggle.com/competitions/aptos2019-blindness-detection (`train.csv`)
- IDRiD, Disease Grading subset, https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid (training and testing label CSV files)
- Messidor (original release), https://www.adcis.net/en/third-party/messidor/ (`Annotation_Base*.xls`)

The APTOS validation split is the stratified 20% split of `train.csv` with `random_state=42`. IDRiD test images are identified with the suffix `test` (e.g., `IDRiD_001test`).

## Reproducing the results

```bash
pip install -r requirements.txt
python code/analysis/merge_labels.py --aptos <aptos>/train.csv \
    --idrid-train "<idrid>/a. IDRiD_Disease Grading_Training Labels.csv" \
    --idrid-test  "<idrid>/b. IDRiD_Disease Grading_Testing Labels.csv" \
    --messidor <messidor_annotation_folder> --pred predictions --out predictions_full
python code/analysis/reproduce_results.py --pred predictions_full --out results_reproduced
DR_ROOT=. DR_RESULTS=results_reproduced DR_PRED=predictions_full python code/analysis/make_tables.py
DR_ROOT=. DR_RESULTS=results_reproduced DR_PRED=predictions_full python code/analysis/make_figures.py
```

`results_reproduced/` is identical to `results/`, and the tables and figures are written to `outputs/`.
The bootstrap confidence intervals use 1,000 resamples with a fixed seed and are therefore reproduced exactly.

## Trained models

The trained Keras models (about 6 GB) are not included and are available from the corresponding author upon reasonable request.

## Licenses

Code is released under the MIT License (`LICENSE`). Predictions and result tables are released under CC BY 4.0 (`LICENSE-DATA`).
The datasets themselves remain under the licenses of their providers.

## How to cite

If you use this code or the predictions, please cite the archived release:

> Kurniawan, I., Avi, P. P., Khoerunnisa, G., Setiawan, H., Md Noar, N. A. Z., & Muthuvalu, M. S. (2026).
> Code and per-image predictions for "Evaluating Cross-Dataset Generalization of Deep Learning Models for Diabetic
> Retinopathy Detection Using Retinal Fundus Images" (Version v1.0). Zenodo. https://doi.org/10.5281/zenodo.23681313

The DOI 10.5281/zenodo.23681313 refers to version v1.0, which corresponds to the article. The concept DOI
10.5281/zenodo.23681312 always resolves to the latest version.

## Contact

Isman Kurniawan, School of Computing, Telkom University, ismankrn@telkomuniversity.ac.id
