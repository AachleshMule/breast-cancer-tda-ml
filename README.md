# Persistence-image topological machine learning for breast cancer detection

Real, reproducible pipeline for the MammoMinds project (MIT-ADT University, TYAIA212).

```
image -> ROI crop -> resize 256x256 (bilinear) -> intensities rescaled to [0, 100] (101 integer levels)
      -> cubical sublevel-set persistent homology, H0 and H1 (Giotto-TDA)
      -> vectorization: Betti curves | persistence landscapes | silhouettes | persistence images
      -> classifiers: KNN, SVM, Random Forest, XGBoost (grid search, 5-fold CV)
      -> accuracy, precision, recall, F1, AUC, homology ablation, figures
```

Everything in `results/` is computed from the images you provide. Nothing is simulated.

> **Never report numbers from the synthetic demo (`make_synthetic_demo.py`).** It exists only to check that the
> code runs. The only numbers allowed in the paper, report and presentation are the ones produced on the real
> CBIS-DDSM and BUSI images.

## 1. Setup (Python 3.10, 3.11 or 3.12)

`giotto-tda` does not support Python 3.13/3.14. If your default Python is 3.14, install 3.12 first
(for example `brew install python@3.12`) and create the environment with it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q            # 14 tests, uses only generated images; all must pass
```

## 2. Datasets

* **BUSI** (Al-Dhabyani et al., Data in Brief 2020): download "Dataset_BUSI_with_GT" (folders `benign`, `malignant`,
  `normal`, each image has `*_mask*.png`). Original source: the authors' page (Cairo University) and the Kaggle
  mirror "Breast Ultrasound Images Dataset".
* **CBIS-DDSM** (Lee et al., Scientific Data 2017): The Cancer Imaging Archive, collection "CBIS-DDSM".
  You need the **ROI / cropped mass images** (DICOM) converted to PNG and the pathology label of each lesion
  (`BENIGN`, `BENIGN_WITHOUT_CALLBACK`, `MALIGNANT`) from `mass_case_description_*_set.csv`.
  Put the PNGs in `benign/` and `malignant/` folders (use `folder` mode) or build a CSV that joins file names and
  labels (use `csv` mode). Keep the patient id (for example `P_00001`) so the split can be patient-level.

## 3. Build the manifests

```bash
# BUSI (normal images are skipped; binary benign vs malignant)
python scripts/build_manifest.py busi --root /path/to/Dataset_BUSI_with_GT --out manifests/busi.csv

# CBIS-DDSM ROI crops already sorted into benign/ and malignant/ (patient id parsed from the file name)
python scripts/build_manifest.py folder --root /path/to/cbis_roi_png --out manifests/cbis.csv \
       --patient-regex "(P_\d+)"
```

A manifest is a CSV with `image_path,label[,mask_path][,patient_id]`. Paths are relative to `--root`.
BUSI masks are used to crop strictly to the lesion bounding box.

## 4. Run

Quick check on a small subset first (a few minutes):

```bash
python scripts/run_experiments.py --manifest manifests/busi.csv --root /path/to/Dataset_BUSI_with_GT \
       --name busi_trial --max-images 200 --fast
```

Full runs (hours, depending on the computer; 16 GB RAM recommended):

```bash
python scripts/run_experiments.py --manifest manifests/busi.csv --root /path/to/Dataset_BUSI_with_GT --name busi
python scripts/run_experiments.py --manifest manifests/cbis.csv --root /path/to/cbis_roi_png        --name cbis
```

If memory is a problem add `--min-persistence 1` (drops zero-width noise features). If you use it, say so in the paper.

## 5. Outputs (per dataset, in `results/<name>/`)

| File | Paper/report item |
|---|---|
| `tables/table3_classifiers.csv` | Table 3 - KNN, Random Forest, SVM, XGBoost on persistence images |
| `tables/table4_vectorizations.csv` | Table 4 - Betti curves, landscapes, silhouettes, persistence images (XGBoost) |
| `tables/table5_ablation.csv` | Table 5 - H0 only, H1 only, H0 + H1 (mean and std over 5 folds) |
| `tables/table6_extended_metrics.csv` | Table 6 - accuracy, precision, recall, F1, AUC (hold-out test set) |
| `tables/table7_hyperparameters.csv`, `table_pi_selection.csv` | Table 7 - tuned values |
| `figures/fig_persistence_diagrams.png` | Persistence diagrams of a real benign and a real malignant image |
| `figures/fig_persistence_images_example.png`, `fig_median_persistence_images.png` | Persistence image figures |
| `figures/fig_roc.png`, `fig_confusion_xgboost.png` | ROC curves, confusion matrix |
| `figures/fig_vectorization_comparison.png`, `fig_ablation.png` | Bar charts |
| `summary.md`, `run_log.json`, `tables/split.csv` | One-page summary, software versions, seed, counts, exact split |

## 6. Methodological notes (state these in the paper)

* **Split:** stratified 80/20 hold-out. When `patient_id` is present the split and all CV folds are patient-grouped
  (`StratifiedGroupKFold`); BUSI has no patient ids, so its split is image-level (say so in the paper).
* **Tuning:** stage 1 chooses the persistence-image bandwidth and grid size (sigma in {0.5,1,1.5,2}, grid in
  {10,20,25}) by 5-fold CV with a default Random Forest on the training set; stage 2 runs the classifier grid
  search of Table 7 with 5-fold CV on the training set. The test set is used once, for reporting.
* **Persistence images:** Giotto-TDA works in (birth, persistence) coordinates with weight = persistence. The paper's
  Eq. (1) is written in (birth, death); change it to (b_i, d_i - b_i) or explain the coordinate change.
* **Leakage control:** vectorizer ranges are fitted on training diagrams only. The reported mean +- std in Tables 4 and 5
  come from pipelines that refit the vectorizer inside every training fold.
* **Essential class:** Giotto-TDA's `reduced_homology=True` (default) removes the one infinite H0 class per image.
* **Feature sizes:** persistence image 2x(n_bins)^2 (800 for 20x20), Betti curves 200, silhouettes 200,
  landscapes 1000 (5 layers x 100 bins x 2 dimensions).
* **Seed:** 42 everywhere (see `run_log.json`).

## 7. Tests

`pytest -q` runs 14 tests (normalization, ROI cropping, preprocessing, diagram validity, feature sizes, dimension
selection, leakage-free split, metrics, manifest parsing, and a complete end-to-end run). Keep the terminal output as
evidence for the Software Testing chapter.
