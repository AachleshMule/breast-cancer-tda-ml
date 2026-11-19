"""
Run the full experiment on ONE dataset and write every table/figure used in the paper and report.

Example:
    python scripts/run_experiments.py --manifest manifests/busi.csv --root /path/to/Dataset_BUSI_with_GT --name busi
    python scripts/run_experiments.py --manifest manifests/cbis.csv --root /path/to/cbis_roi_png     --name cbis

Add --fast for a quick smoke test (tiny grids). Add --max-images 300 to try the pipeline on a subset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plots  # noqa: E402
import tda_pipeline as tp  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def fit_eval(D_tr, y_tr, D_te, y_te, dims, vec_name, clf_name, params, vec_kw):
    """Fit vectorizer+classifier on the training diagrams, evaluate on the held-out test diagrams."""
    pipe = Pipeline([("vec", tp.make_vectorizer(vec_name, **vec_kw)),
                     ("flat", FunctionTransformer(tp.flatten, validate=False)),
                     ("clf", tp.make_classifier(clf_name, **params))])
    pipe.fit(tp.select_dims(D_tr, dims), y_tr)
    Dte = tp.select_dims(D_te, dims)
    pred = pipe.predict(Dte)
    sc = pipe.predict_proba(Dte)[:, 1]
    return tp.metrics(y_te, pred, sc), pipe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--name", required=True, help="dataset name used for the output folder, e.g. busi or cbis")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=tp.IMG_SIZE)
    ap.add_argument("--fast", action="store_true", help="tiny grids, for smoke tests only")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--min-persistence", type=float, default=0.0,
                    help="drop features with persistence <= this value (0 keeps everything, as in the paper)")
    a = ap.parse_args(argv)

    t0 = time.time()
    out = Path(a.out) / a.name
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    log = {"environment": tp.environment_info(), "args": vars(a)}

    # ---- data -------------------------------------------------------------------
    manifest = tp.load_manifest(a.manifest, a.root)
    X, y, groups, manifest = tp.load_dataset(manifest, a.size, a.max_images, out / "cache" / "images.npz")
    log["n_images"] = int(len(y))
    log["n_benign"] = int((y == 0).sum())
    log["n_malignant"] = int((y == 1).sum())
    log["patient_grouped"] = groups is not None
    print(f"[data] {len(y)} images  benign={log['n_benign']}  malignant={log['n_malignant']}  "
          f"patient-grouped={groups is not None}")

    # ---- persistent homology ----------------------------------------------------
    t1 = time.time()
    D = tp.compute_diagrams(X, a.n_jobs, a.min_persistence)
    log["diagram_shape"] = list(D.shape)
    log["seconds_persistent_homology"] = round(time.time() - t1, 1)
    print(f"[ph] diagrams {D.shape} in {log['seconds_persistent_homology']} s")

    tr, te = tp.holdout_split(y, groups)
    pd.DataFrame({"index": np.concatenate([tr, te]),
                  "split": ["train"] * len(tr) + ["test"] * len(te)}).to_csv(out / "tables" / "split.csv", index=False)
    if groups is not None:
        assert not set(groups[tr]) & set(groups[te]), "patient leakage between train and test"
    D_tr, D_te, y_tr, y_te = D[tr], D[te], y[tr], y[te]
    g_tr = groups[tr] if groups is not None else None
    log["n_train"], log["n_test"] = int(len(tr)), int(len(te))

    # ---- persistence-image configuration (stage 1) and classifiers (stage 2) -----
    pi_cfg, pi_tab = tp.select_pi_config(D_tr, y_tr, g_tr, a.fast, a.n_jobs)
    pi_tab.to_csv(out / "tables" / "table_pi_selection.csv", index=False)
    print(f"[pi] selected {pi_cfg}")
    vec = tp.make_vectorizer("pi", n_jobs=a.n_jobs, **pi_cfg)
    P_tr = vec.fit_transform(D_tr)
    P_te = vec.transform(D_te)
    F_tr, F_te = tp.flatten(P_tr), tp.flatten(P_te)
    log["feature_dim"] = int(F_tr.shape[1])

    models, best_params, rows3, rows6 = {}, {}, [], []
    for name, label in (("knn", "KNN"), ("rf", "Random Forest"), ("svm", "SVM"), ("xgb", "XGBoost")):
        est, best, cv_acc = tp.tune_classifier(name, F_tr, y_tr, g_tr, a.fast)
        pred, sc = est.predict(F_te), tp.score_of(est, F_te)
        m = tp.metrics(y_te, pred, sc)
        models[name], best_params[name] = est, best
        rows3.append({"Classifier": label, "Test accuracy (%)": round(100 * m["accuracy"], 2),
                      "Train CV accuracy (%)": round(100 * cv_acc, 2)})
        if name in ("rf", "xgb"):
            rows6.append({"Classifier": label, **{k: round(v, 4) for k, v in m.items()}})
        print(f"[clf] {label:14s} test acc {100 * m['accuracy']:.2f}%  best={best}")
    t3 = pd.DataFrame(rows3)
    t6 = pd.DataFrame(rows6)
    t3.to_csv(out / "tables" / "table3_classifiers.csv", index=False)
    t6.to_csv(out / "tables" / "table6_extended_metrics.csv", index=False)

    xgb_best = best_params["xgb"]
    t7 = pd.DataFrame([{"Model": "Random Forest", "Selected": json.dumps(best_params["rf"])},
                       {"Model": "XGBoost", "Selected": json.dumps(xgb_best)},
                       {"Model": "KNN", "Selected": json.dumps(best_params["knn"])},
                       {"Model": "SVM", "Selected": json.dumps(best_params["svm"])},
                       {"Model": "Persistence image", "Selected": json.dumps(pi_cfg)}])
    t7.to_csv(out / "tables" / "table7_hyperparameters.csv", index=False)

    # ---- vectorization comparison (XGBoost with the tuned parameters) -----------
    rows4 = []
    for vname, label in (("betti", "Betti curves"), ("landscape", "Persistence landscapes"),
                         ("silhouette", "Silhouettes"), ("pi", "Persistence images")):
        vkw = dict(pi_cfg) if vname == "pi" else {}
        m, _ = fit_eval(D_tr, y_tr, D_te, y_te, (0, 1), vname, "xgb", xgb_best, dict(vkw, n_jobs=a.n_jobs))
        cv = tp.cv_pipeline_scores(D, y, groups, (0, 1), vname, "xgb", xgb_best, vkw, a.n_jobs)
        rows4.append({"Vectorization": label, "Test accuracy (%)": round(100 * m["accuracy"], 2),
                      "CV accuracy mean (%)": round(100 * cv["acc_mean"], 2),
                      "CV accuracy std (%)": round(100 * cv["acc_std"], 2)})
        print(f"[vec] {label:24s} test {100 * m['accuracy']:.2f}%  cv {100 * cv['acc_mean']:.2f} +- {100 * cv['acc_std']:.2f}")
    t4 = pd.DataFrame(rows4)
    t4.to_csv(out / "tables" / "table4_vectorizations.csv", index=False)

    # ---- homology-dimension ablation -------------------------------------------
    rows5 = []
    for dims, label in (((0,), "H0 only"), ((1,), "H1 only"), ((0, 1), "H0 + H1")):
        m, _ = fit_eval(D_tr, y_tr, D_te, y_te, dims, "pi", "xgb", xgb_best, dict(pi_cfg, n_jobs=a.n_jobs))
        cv = tp.cv_pipeline_scores(D, y, groups, dims, "pi", "xgb", xgb_best, pi_cfg, a.n_jobs)
        rows5.append({"Features used": label, "Test accuracy (%)": round(100 * m["accuracy"], 2),
                      "CV accuracy mean (%)": round(100 * cv["acc_mean"], 2),
                      "CV accuracy std (%)": round(100 * cv["acc_std"], 2)})
        print(f"[abl] {label:8s} test {100 * m['accuracy']:.2f}%  cv {100 * cv['acc_mean']:.2f} +- {100 * cv['acc_std']:.2f}")
    t5 = pd.DataFrame(rows5)
    t5.to_csv(out / "tables" / "table5_ablation.csv", index=False)

    # ---- figures ----------------------------------------------------------------
    fig = out / "figures"
    i_b = int(np.flatnonzero(y == 0)[0])
    i_m = int(np.flatnonzero(y == 1)[0])
    plots.plot_persistence_diagrams(D[i_b], D[i_m], fig / "fig_persistence_diagrams.png")
    two = vec.transform(D[[i_b, i_m]])
    plots.plot_pi_pair(two[0], two[1], fig / "fig_persistence_images_example.png")
    plots.plot_median_pi(np.median(P_tr[y_tr == 0], axis=0), np.median(P_tr[y_tr == 1], axis=0),
                         fig / "fig_median_persistence_images.png")
    curves = []
    for name, label in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        sc = tp.score_of(models[name], F_te)
        fpr, tpr, _ = roc_curve(y_te, sc)
        curves.append((label, fpr, tpr, tp.metrics(y_te, models[name].predict(F_te), sc)["auc"]))
    plots.plot_roc(curves, fig / "fig_roc.png")
    plots.plot_confusion(confusion_matrix(y_te, models["xgb"].predict(F_te)), fig / "fig_confusion_xgboost.png")
    plots.plot_bars(list(t4["Vectorization"].str.replace(" ", "\n", n=1)), list(t4["CV accuracy mean (%)"]),
                    list(t4["CV accuracy std (%)"]), fig / "fig_vectorization_comparison.png",
                    colors=["#9aa5b1"] * 3 + ["#1f5fa8"])
    plots.plot_bars(list(t5["Features used"]), list(t5["CV accuracy mean (%)"]), list(t5["CV accuracy std (%)"]),
                    fig / "fig_ablation.png")

    # ---- summary ----------------------------------------------------------------
    log["seconds_total"] = round(time.time() - t0, 1)
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    summary = ["# Results for dataset: " + a.name, "",
               f"Images: {log['n_images']} (benign {log['n_benign']}, malignant {log['n_malignant']}); "
               f"train {log['n_train']} / test {log['n_test']}; patient-grouped split: {log['patient_grouped']}.", "",
               "## Table 3 - classifiers (persistence images)", md_table(t3), "",
               "## Table 4 - vectorizations (XGBoost)", md_table(t4), "",
               "## Table 5 - homology ablation (persistence images, XGBoost)", md_table(t5), "",
               "## Table 6 - extended metrics (hold-out test set)", md_table(t6), "",
               "## Table 7 - selected hyperparameters", md_table(t7), "",
               "## Persistence-image configuration search", md_table(pi_tab), ""]
    (out / "summary.md").write_text("\n".join(summary))
    print(f"[done] {log['seconds_total']} s  ->  {out}")
    return out


if __name__ == "__main__":
    main()
