"""
Persistence-image topological machine learning for breast cancer detection.

Real pipeline (no simulated data):
  image -> ROI crop -> resize 256x256 (bilinear) -> intensities in [0, 100]
        -> cubical sublevel-set persistent homology (Giotto-TDA, H0 and H1)
        -> vectorization (Betti curves / landscapes / silhouettes / persistence images)
        -> classifiers (KNN, SVM, Random Forest, XGBoost)
        -> accuracy, precision, recall, F1, AUC, ablation, figures.

Every number written to the results folder is computed from the images you give it.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
IMG_SIZE = 256


# ----------------------------------------------------------------------------
# Data loading and preprocessing
# ----------------------------------------------------------------------------
def read_gray(path) -> np.ndarray:
    """Read an image as a float32 grayscale array (handles 8-bit and 16-bit)."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return np.asarray(im).astype(np.float32)
    return np.asarray(im.convert("L")).astype(np.float32)


def crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop strictly to the bounding box of the (non-zero) ROI mask."""
    if mask.shape != img.shape:
        m = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask = np.asarray(m.resize((img.shape[1], img.shape[0]), Image.NEAREST))
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:  # empty mask -> keep the full image
        return img
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def resize_bilinear(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    im = Image.fromarray(img.astype(np.float32), mode="F")
    return np.asarray(im.resize((size, size), Image.BILINEAR)).astype(np.float32)


def normalize_0_100(img: np.ndarray) -> np.ndarray:
    """Min-max renormalize to [0, 100] and round (integer filtration levels 0..100)."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.round((img - lo) / (hi - lo) * 100.0).astype(np.float32)


def preprocess_image(path, mask_paths=(), size: int = IMG_SIZE) -> np.ndarray:
    img = read_gray(path)
    if mask_paths:
        union = None
        for mp in mask_paths:
            m = read_gray(mp)
            if m.shape != img.shape:
                m = np.asarray(Image.fromarray(m, mode="F").resize((img.shape[1], img.shape[0]), Image.NEAREST))
            union = (m > 0) if union is None else (union | (m > 0))
        img = crop_to_mask(img, union.astype(np.uint8))
    return normalize_0_100(resize_bilinear(img, size))


def parse_label(v):
    s = str(v).strip().lower()
    if s in ("1", "malignant") or s.startswith("malignant"):
        return 1
    if s in ("0", "benign") or s.startswith("benign"):
        return 0
    return None  # e.g. 'normal' -> excluded from binary classification


def load_manifest(csv_path, root=None) -> pd.DataFrame:
    """Manifest columns: image_path, label, [mask_path], [patient_id].
    mask_path may hold several masks separated by ';'. Paths are relative to root."""
    df = pd.read_csv(csv_path)
    for c in ("image_path", "label"):
        if c not in df.columns:
            raise ValueError(f"manifest is missing column '{c}'")
    df["y"] = df["label"].map(parse_label)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    root = Path(root) if root else Path(".")
    df["abs_image"] = [str(root / p) for p in df["image_path"]]
    if "mask_path" in df.columns:
        df["abs_masks"] = [
            [str(root / m.strip()) for m in str(p).split(";") if m.strip() and m.strip().lower() != "nan"]
            for p in df["mask_path"]
        ]
    else:
        df["abs_masks"] = [[] for _ in range(len(df))]
    if "patient_id" in df.columns:
        pid = df["patient_id"].astype(object).copy()
        missing = pid.isna() | (pid.astype(str).str.strip() == "")
        for i in np.flatnonzero(missing.values):
            pid.iloc[i] = f"_row{i}"
        df["group"] = pid.astype(str)
    else:
        df["group"] = None
    return df.reset_index(drop=True)


def load_dataset(manifest: pd.DataFrame, size: int = IMG_SIZE, max_images=None, cache=None, seed=SEED):
    """Returns X (n, size, size) float32, y, groups (array or None), kept manifest rows."""
    if max_images and len(manifest) > max_images:  # stratified subsample for quick trials
        parts = []
        for _, g in manifest.groupby("y"):
            k = max(1, int(round(max_images * len(g) / len(manifest))))
            parts.append(g.sample(min(len(g), k), random_state=seed))
        manifest = pd.concat(parts).sort_index().reset_index(drop=True)
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["y"]) == len(manifest) and list(z["paths"]) == list(manifest["image_path"]):
            groups = z["groups"] if z["groups"].size else None
            return z["X"], z["y"], groups, manifest
    X = np.stack([preprocess_image(r.abs_image, r.abs_masks, size) for r in manifest.itertuples()])
    y = manifest["y"].to_numpy()
    groups = manifest["group"].to_numpy() if manifest["group"].notna().all() else None
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, paths=manifest["image_path"].to_numpy(),
                            groups=groups if groups is not None else np.array([]))
    return X, y, groups, manifest


# ----------------------------------------------------------------------------
# Persistent homology and vectorization (Giotto-TDA)
# ----------------------------------------------------------------------------
def compute_diagrams(X: np.ndarray, n_jobs=-1, min_persistence: float = 0.0) -> np.ndarray:
    """Cubical sublevel-set persistence, homology dimensions 0 and 1.
    Output: (n, n_points, 3) array of [birth, death, dimension]."""
    from gtda.homology import CubicalPersistence
    D = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=n_jobs).fit_transform(X)
    if min_persistence > 0:
        D = select_dims(D, (0, 1), min_persistence)
    return D


def select_dims(D: np.ndarray, dims=(0, 1), min_persistence: float = 0.0) -> np.ndarray:
    """Keep only the requested homology dimensions (and optionally drop tiny features).
    Giotto-TDA needs every diagram to hold the same number of points in each dimension,
    so each dimension is padded with zero-persistence points (0, 0, q) up to its maximum count."""
    per_dim = {int(q): [] for q in dims}
    for d in D:
        for q in per_dim:
            rows = d[(d[:, 2] == q) & ((d[:, 1] - d[:, 0]) > min_persistence)]
            per_dim[q].append(rows)
    blocks = []
    for q, lst in per_dim.items():
        L = max(1, max(len(r) for r in lst))
        arr = np.zeros((len(lst), L, 3), dtype=np.float64)
        arr[:, :, 2] = q
        for i, r in enumerate(lst):
            arr[i, : len(r)] = r
        blocks.append(arr)
    return np.concatenate(blocks, axis=1)


def persistence_weight(p):
    """Weight = persistence, as in the paper (w_i = d_i - b_i)."""
    return p


def flatten(X):
    return np.asarray(X).reshape(len(X), -1)


def make_vectorizer(name: str, sigma: float = 1.0, n_bins: int = 20, n_jobs=-1):
    from gtda.diagrams import BettiCurve, PersistenceImage, PersistenceLandscape, Silhouette
    if name == "pi":
        return PersistenceImage(sigma=sigma, n_bins=n_bins, weight_function=persistence_weight, n_jobs=n_jobs)
    if name == "betti":
        return BettiCurve(n_bins=100, n_jobs=n_jobs)                        # 2 x 100 = 200 features
    if name == "landscape":
        return PersistenceLandscape(n_layers=5, n_bins=100, n_jobs=n_jobs)   # 2 x 5 x 100 = 1000
    if name == "silhouette":
        return Silhouette(power=1.0, n_bins=100, n_jobs=n_jobs)              # 2 x 100 = 200
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Models, splits, metrics
# ----------------------------------------------------------------------------
GRIDS = {
    "rf": {"n_estimators": [50, 100, 150, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
    "xgb": {"learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 6, 9], "n_estimators": [100, 200, 300]},
    "svm": {"C": [0.1, 1, 10]},
    "knn": {"n_neighbors": [3, 5, 7]},
}
FAST_GRIDS = {
    "rf": {"n_estimators": [50, 100], "max_depth": [None, 10], "min_samples_split": [2]},
    "xgb": {"learning_rate": [0.1], "max_depth": [3, 6], "n_estimators": [100]},
    "svm": {"C": [1]},
    "knn": {"n_neighbors": [5]},
}
PI_GRID = {"sigma": [0.5, 1.0, 1.5, 2.0], "n_bins": [10, 20, 25]}


def make_classifier(name: str, **params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if name == "xgb":
        return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, tree_method="hist", **params)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED, **params))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))
    raise ValueError(name)


def _prefix(name, grid):
    """Pipeline-wrapped models (svm, knn) need step-prefixed parameter names."""
    if name == "svm":
        return {f"svc__{k}": v for k, v in grid.items()}
    if name == "knn":
        return {f"kneighborsclassifier__{k}": v for k, v in grid.items()}
    return grid


def holdout_split(y, groups, seed=SEED):
    """Stratified 80/20 split; grouped by patient when groups are available."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    idx = np.arange(len(y))
    if groups is not None:
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        tr, te = next(sg.split(idx, y, groups))
        return tr, te
    return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)


def cv_splits(y, groups, n_splits=5, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    idx = np.zeros(len(y))
    if groups is not None:
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, y))


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    ok = y_score is not None and len(np.unique(y_true)) == 2
    out["auc"] = roc_auc_score(y_true, y_score) if ok else float("nan")
    return out


def select_pi_config(D_train, y_train, g_train, fast=False, n_jobs=-1):
    """Stage 1: choose (sigma, grid size) for the persistence image by 5-fold CV with a default Random Forest.
    The vectorizer ranges are fitted on the training images only (no labels, no test images)."""
    rows, best = [], None
    grid = {"sigma": [1.0], "n_bins": [20]} if fast else PI_GRID
    splits = cv_splits(y_train, g_train)
    for s in grid["sigma"]:
        for b in grid["n_bins"]:
            F = flatten(make_vectorizer("pi", sigma=s, n_bins=b, n_jobs=n_jobs).fit_transform(D_train))
            accs = []
            for tr, va in splits:
                clf = make_classifier("rf", n_estimators=100)
                clf.fit(F[tr], y_train[tr])
                accs.append(float((clf.predict(F[va]) == y_train[va]).mean()))
            rows.append({"sigma": s, "n_bins": b, "cv_accuracy": float(np.mean(accs)), "cv_std": float(np.std(accs))})
            if best is None or rows[-1]["cv_accuracy"] > best[0]:
                best = (rows[-1]["cv_accuracy"], s, b)
    return {"sigma": best[1], "n_bins": best[2]}, pd.DataFrame(rows)


def tune_classifier(name, F_train, y_train, g_train, fast=False):
    """Stage 2: grid search with 5-fold (patient-grouped) CV on the training set."""
    from sklearn.model_selection import GridSearchCV
    grid = _prefix(name, (FAST_GRIDS if fast else GRIDS)[name])
    gs = GridSearchCV(make_classifier(name), grid, cv=cv_splits(y_train, g_train), scoring="accuracy",
                      n_jobs=1, refit=True)
    gs.fit(F_train, y_train)
    best = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    return gs.best_estimator_, best, float(gs.best_score_)


def cv_pipeline_scores(D, y, groups, dims, vec_name, clf_name, clf_params, vec_kw, n_jobs=-1):
    """5-fold cross-validation with the vectorizer fitted inside every training fold."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    Dsel = select_dims(D, dims)
    accs, f1s, aucs = [], [], []
    for tr, va in cv_splits(y, groups):
        pipe = Pipeline([("vec", make_vectorizer(vec_name, n_jobs=n_jobs, **vec_kw)),
                         ("flat", FunctionTransformer(flatten, validate=False)),
                         ("clf", make_classifier(clf_name, **clf_params))])
        pipe.fit(Dsel[tr], y[tr])
        pred = pipe.predict(Dsel[va])
        sc = pipe.predict_proba(Dsel[va])[:, 1]
        m = metrics(y[va], pred, sc)
        accs.append(m["accuracy"]); f1s.append(m["f1"]); aucs.append(m["auc"])
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "auc_mean": float(np.nanmean(aucs)),
            "folds": [float(a) for a in accs]}


def environment_info() -> dict:
    import gtda
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "giotto-tda": gtda.__version__, "seed": SEED, "image_size": IMG_SIZE,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
