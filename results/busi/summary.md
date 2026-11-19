# Results for dataset: busi

Images: 647 (benign 437, malignant 210); train 517 / test 130; patient-grouped split: True.

## Table 3 - classifiers (persistence images)
| Classifier | Test accuracy (%) | Train CV accuracy (%) |
|---|---|---|
| KNN | 67.6900 | 74.6500 |
| Random Forest | 84.6200 | 82.1900 |
| SVM | 79.2300 | 80.0700 |
| XGBoost | 86.9200 | 82.3900 |

## Table 4 - vectorizations (XGBoost)
| Vectorization | Test accuracy (%) | CV accuracy mean (%) | CV accuracy std (%) |
|---|---|---|---|
| Betti curves | 86.9200 | 82.5300 | 2.8900 |
| Persistence landscapes | 83.0800 | 81.7600 | 3.4700 |
| Silhouettes | 81.5400 | 79.7500 | 2.5000 |
| Persistence images | 86.9200 | 82.9900 | 2.9900 |

## Table 5 - homology ablation (persistence images, XGBoost)
| Features used | Test accuracy (%) | CV accuracy mean (%) | CV accuracy std (%) |
|---|---|---|---|
| H0 only | 83.0800 | 80.9900 | 2.9300 |
| H1 only | 76.9200 | 78.5200 | 1.8900 |
| H0 + H1 | 86.9200 | 82.9900 | 2.9900 |

## Table 6 - extended metrics (hold-out test set)
| Classifier | accuracy | precision | recall | f1 | auc |
|---|---|---|---|---|---|
| Random Forest | 0.8462 | 0.7692 | 0.7317 | 0.7500 | 0.9183 |
| XGBoost | 0.8692 | 0.8158 | 0.7561 | 0.7848 | 0.9216 |

## Table 7 - selected hyperparameters
| Model | Selected |
|---|---|
| Random Forest | {"max_depth": 10, "min_samples_split": 2, "n_estimators": 150} |
| XGBoost | {"learning_rate": 0.05, "max_depth": 9, "n_estimators": 100} |
| KNN | {"n_neighbors": 7} |
| SVM | {"C": 1} |
| Persistence image | {"sigma": 1.5, "n_bins": 20} |

## Persistence-image configuration search
| sigma | n_bins | cv_accuracy | cv_std |
|---|---|---|---|
| 0.5000 | 10.0000 | 0.7735 | 0.0369 |
| 0.5000 | 20.0000 | 0.8045 | 0.0505 |
| 0.5000 | 25.0000 | 0.8083 | 0.0580 |
| 1.0000 | 10.0000 | 0.7735 | 0.0369 |
| 1.0000 | 20.0000 | 0.7968 | 0.0349 |
| 1.0000 | 25.0000 | 0.8064 | 0.0394 |
| 1.5000 | 10.0000 | 0.7735 | 0.0369 |
| 1.5000 | 20.0000 | 0.8181 | 0.0231 |
| 1.5000 | 25.0000 | 0.8180 | 0.0419 |
| 2.0000 | 10.0000 | 0.7929 | 0.0285 |
| 2.0000 | 20.0000 | 0.8123 | 0.0264 |
| 2.0000 | 25.0000 | 0.8161 | 0.0359 |
