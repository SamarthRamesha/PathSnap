"""Bare-minimum model experiment harness for PathSnap candidate classification."""

from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Single variable for table and data paths
PATHS = {
    "train": Path("full_features/train_candidate_features_sample.csv"),
    "test": Path("full_features/student_test_candidate_features_sample.csv"),
    "test_data_dir": Path("PathSnap/data/student_test"),
}

SEED = 42
KNN_MAX_TRAIN = 200_000  # shortcut: cap KNN fit rows for speed on 2M+ row tables; remove if using ANN index
NON_FEATURE_COLS = {"trip_id", "point_idx", "arc_id", "label"}
DROP_GEO_TIME = {"longitude", "latitude", "timestamp"}
RESULTS_CSV = Path("results.csv")
PLOT_PNG = Path("model_comparison.png")


def build_models():
    """Return (name, family, config_str, estimator) for 5 model families."""
    return [
        (
            "LogisticRegression",
            "Linear",
            "StandardScaler + max_iter=1000",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]),
        ),
        (
            "KNN",
            "Instance-based",
            f"StandardScaler + k=11 (max {KNN_MAX_TRAIN // 1000}k train subsample)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=11, n_jobs=-1)),
            ]),
        ),
        (
            "RandomForest",
            "Ensemble",
            "n_estimators=100, max_depth=15",
            RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1
            ),
        ),
        (
            "LightGBM",
            "Boosting",
            "n_estimators=200, lr=0.05",
            LGBMClassifier(
                n_estimators=200, learning_rate=0.05, random_state=SEED,
                n_jobs=-1, verbose=-1
            ),
        ),
        (
            "MLP",
            "Neural Network",
            "StandardScaler + (64,32), max_iter=500, batch=1024, early_stopping",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500, batch_size=1024,
                    early_stopping=True, random_state=SEED
                )),
            ]),
        ),
    ]


def top1_accuracy(df, probs):
    """Per (trip_id, point_idx), pick highest-prob candidate and check label == 1."""
    tmp = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "point_idx": df["point_idx"].to_numpy(),
        "label": df["label"].to_numpy(),
        "prob": probs,
    })
    best_idx = tmp.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    return float(tmp.loc[best_idx, "label"].mean())


def load_trip_arcs(trip_ids, data_dir):
    """Load each trip's .arcs file as an (N, 2) array of [from_node, to_node]."""
    arcs = {}
    for tid in trip_ids:
        tid_str = f"{int(tid):08d}"
        path = data_dir / tid_str / f"{tid_str}.arcs"
        arcs[int(tid)] = np.atleast_2d(np.loadtxt(path, dtype=int))
    return arcs


def seq_continuity(arc_seq, nodes):
    """Fraction of consecutive unique arcs where to_node(a1) == from_node(a2)."""
    if len(arc_seq) == 0:
        return 0.0
    # Collapse consecutive duplicate arcs
    collapsed = [int(arc_seq[0])] + [
        int(a) for a, prev in zip(arc_seq[1:], arc_seq[:-1]) if a != prev
    ]
    if len(collapsed) < 2:
        return 1.0
    connected = sum(
        nodes[a1, 1] == nodes[a2, 0]
        for a1, a2 in zip(collapsed[:-1], collapsed[1:])
        if 0 <= a1 < len(nodes) and 0 <= a2 < len(nodes)
    )
    return connected / (len(collapsed) - 1)


def route_continuity(df, probs, arcs_by_trip):
    """Mean per-trip continuity of top-1 predicted arcs on student-test."""
    tmp = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "point_idx": df["point_idx"].to_numpy(),
        "arc_id": df["arc_id"].to_numpy(),
        "prob": probs,
    })
    best = tmp.loc[tmp.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()]

    trip_scores = []
    for tid, grp in best.groupby("trip_id"):
        seq = grp.sort_values("point_idx")["arc_id"].to_numpy()
        trip_scores.append(seq_continuity(seq, arcs_by_trip[int(tid)]))

    score = float(np.mean(trip_scores))
    assert 0.0 <= score <= 1.0, f"Invalid route continuity: {score}"
    return score


def reference_route_continuity(trip_ids, data_dir, arcs_by_trip):
    """Mean per-trip continuity of the reference .route files using the exact same metric."""
    trip_scores = []
    for tid in trip_ids:
        tid_str = f"{int(tid):08d}"
        route_path = data_dir / tid_str / f"{tid_str}.route"
        seq = np.atleast_1d(np.loadtxt(route_path, dtype=int))
        trip_scores.append(seq_continuity(seq, arcs_by_trip[int(tid)]))
    return float(np.mean(trip_scores))


def fit_maybe_subsample(model, X, y, model_name):
    """Fit model; subsample to KNN_MAX_TRAIN rows for KNN if dataset is large."""
    if model_name.startswith("KNN") and len(X) > KNN_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), size=KNN_MAX_TRAIN, replace=False)
        model.fit(X.iloc[idx], y[idx])
        return idx
    model.fit(X, y)
    return None


def evaluate_model(name, family, cfg, est, feature_set, feature_cols,
                   train_df, test_df, gkf, arcs_by_trip, run_ts):
    """Run 5-fold GroupKFold CV on train, then fit on full train and evaluate."""
    print(f"\n[{name}] ({feature_set}, {len(feature_cols)} features)", flush=True)
    X_train = train_df[feature_cols]
    y_train = train_df["label"].to_numpy()
    groups = train_df["trip_id"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["label"].to_numpy()

    # 1) Group-aware CV on training trips only
    cv_f1, cv_top1, cv_ll = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        t_f = time.perf_counter()
        fold_model = clone(est)
        fit_maybe_subsample(fold_model, X_train.iloc[tr_idx], y_train[tr_idx], name)
        va_prob = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        va_pred = (va_prob >= 0.5).astype(int)
        f1 = f1_score(y_train[va_idx], va_pred, average="macro")
        t1 = top1_accuracy(train_df.iloc[va_idx], va_prob)
        ll = log_loss(y_train[va_idx], va_prob)
        cv_f1.append(f1)
        cv_top1.append(t1)
        cv_ll.append(ll)
        print(f"  Fold {fold}/5: F1={f1:.4f}, Top-1={t1:.4f}, LogLoss={ll:.4f} ({time.perf_counter() - t_f:.1f}s)", flush=True)

    # 2) Fit on full train set; score on train (overfitting check) and student-test
    model = clone(est)
    t0 = time.perf_counter()
    sub_idx = fit_maybe_subsample(model, X_train, y_train, name)
    train_time = time.perf_counter() - t0

    # For KNN on 2M+ rows, score train check on a 200k-row trip-aligned slice to avoid multi-minute queries
    if sub_idx is not None:
        # Keep complete trips (~200k rows) so train_top1_acc groups stay intact
        sample_trips = train_df["trip_id"].unique()[:8]
        tr_eval_df = train_df[train_df["trip_id"].isin(sample_trips)]
        tr_prob = model.predict_proba(tr_eval_df[feature_cols])[:, 1]
        tr_y = tr_eval_df["label"].to_numpy()
    else:
        tr_eval_df = train_df
        tr_prob = model.predict_proba(X_train)[:, 1]
        tr_y = y_train
    tr_pred = (tr_prob >= 0.5).astype(int)

    t0 = time.perf_counter()
    te_prob = model.predict_proba(X_test)[:, 1]
    infer_time = time.perf_counter() - t0
    assert np.all((te_prob >= 0.0) & (te_prob <= 1.0)), "Invalid probabilities"
    te_pred = (te_prob >= 0.5).astype(int)

    row = {
        "timestamp": run_ts,
        "train_file": f"full_features/{PATHS['train'].name}",
        "feature_set": feature_set,
        "n_features": len(feature_cols),
        "model": name,
        "family": family,
        "config": cfg,
        "seed": SEED,
        "cv_macro_f1_mean": round(float(np.mean(cv_f1)), 4),
        "cv_macro_f1_std": round(float(np.std(cv_f1)), 4),
        "cv_top1_mean": round(float(np.mean(cv_top1)), 4),
        "cv_top1_std": round(float(np.std(cv_top1)), 4),
        "cv_log_loss_mean": round(float(np.mean(cv_ll)), 4),
        "train_macro_f1": round(float(f1_score(tr_y, tr_pred, average="macro")), 4),
        "train_top1_acc": round(top1_accuracy(tr_eval_df, tr_prob), 4),
        "test_macro_f1": round(float(f1_score(y_test, te_pred, average="macro")), 4),
        # Macro precision/recall match macro F1 across both classes
        "test_precision": round(float(precision_score(y_test, te_pred, average="macro", zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, te_pred, average="macro", zero_division=0)), 4),
        "test_log_loss": round(float(log_loss(y_test, te_prob)), 4),
        "test_top1_acc": round(top1_accuracy(test_df, te_prob), 4),
        "test_route_continuity": round(route_continuity(test_df, te_prob, arcs_by_trip), 4),
        "train_time_s": round(train_time, 3),
        "infer_time_s": round(infer_time, 3),
    }
    print(f"  Full fit={train_time:.2f}s | Test F1={row['test_macro_f1']:.4f}, "
          f"Top-1={row['test_top1_acc']:.4f}, LogLoss={row['test_log_loss']:.4f}, "
          f"RouteCont={row['test_route_continuity']:.4f}, Infer={infer_time:.2f}s", flush=True)
    return row


def make_plot(run_df, ref_cont):
    """Save grouped bar chart for model rows with reference route continuity line."""
    plot_df = run_df[run_df["family"] != "Ground Truth"].reset_index(drop=True)
    x = np.arange(len(plot_df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - 1.5 * w, plot_df["cv_macro_f1_mean"], w, yerr=plot_df["cv_macro_f1_std"],
           capsize=3, label="CV Macro F1")
    ax.bar(x - 0.5 * w, plot_df["test_macro_f1"], w, label="Test Macro F1")
    ax.bar(x + 0.5 * w, plot_df["test_top1_acc"], w, label="Test Top-1 Acc")
    ax.bar(x + 1.5 * w, plot_df["test_route_continuity"], w, label="Test Route Continuity")
    ax.axhline(ref_cont, color="black", linestyle="--", linewidth=1.2,
               label=f"Reference .route Continuity ({ref_cont:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PathSnap Model Comparison (80 Train Trips / 10 Student-Test Trips)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    plt.close(fig)


def main():
    print("Loading full feature tables...", flush=True)
    train_df = pd.read_csv(PATHS["train"])
    test_df = pd.read_csv(PATHS["test"])

    clean_cols = [c for c in train_df.columns if c not in (NON_FEATURE_COLS | DROP_GEO_TIME)]
    full_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    assert NON_FEATURE_COLS.isdisjoint(full_cols), "ID/label leaked into features"
    assert not train_df[full_cols].isna().any().any(), "NaNs in train features"
    assert not test_df[full_cols].isna().any().any(), "NaNs in test features"

    # Downcast feature columns to float32 to cut memory and speed up training
    train_df[full_cols] = train_df[full_cols].astype(np.float32)
    test_df[full_cols] = test_df[full_cols].astype(np.float32)

    n_splits = min(5, train_df["trip_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    test_trips = sorted(test_df["trip_id"].unique())
    arcs_by_trip = load_trip_arcs(test_trips, PATHS["test_data_dir"])
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Train: {train_df.shape} ({train_df['trip_id'].nunique()} trips) | "
          f"Test: {test_df.shape} ({len(test_trips)} trips)", flush=True)
    print(f"Clean features ({len(clean_cols)}): {clean_cols}", flush=True)

    rows = []
    # Baseline row: continuity of the reference .route itself
    ref_cont = reference_route_continuity(test_trips, PATHS["test_data_dir"], arcs_by_trip)
    print(f"\n[Reference (.route)] Student-test route continuity = {ref_cont:.4f}", flush=True)
    rows.append({
        "timestamp": run_ts,
        "train_file": f"full_features/{PATHS['train'].name}",
        "feature_set": "reference_route",
        "n_features": 0,
        "model": "Reference (.route)",
        "family": "Ground Truth",
        "config": "Raw .route arc sequence",
        "seed": SEED,
        "test_route_continuity": round(ref_cont, 4),
    })

    # 5 models on the 13 clean features
    models = build_models()
    for name, family, cfg, est in models:
        rows.append(evaluate_model(
            name, family, cfg, est, "clean_13", clean_cols,
            train_df, test_df, gkf, arcs_by_trip, run_ts
        ))

    # 16-feature ablation (+longitude, latitude, timestamp) for LightGBM and LogisticRegression
    for target_name in ("LogisticRegression", "LightGBM"):
        name, family, cfg, est = [m for m in models if m[0] == target_name][0]
        rows.append(evaluate_model(
            f"{name} (+lon/lat/ts)", family, cfg, est, "with_geo_time_16", full_cols,
            train_df, test_df, gkf, arcs_by_trip, run_ts
        ))

    run_df = pd.DataFrame(rows)

    display_cols = [
        "model", "n_features", "cv_macro_f1_mean", "cv_macro_f1_std",
        "cv_top1_mean", "cv_log_loss_mean", "train_macro_f1", "train_top1_acc",
        "test_macro_f1", "test_top1_acc", "test_log_loss",
        "test_route_continuity", "train_time_s", "infer_time_s",
    ]
    print("\n=== FULL 80/10 RESULTS ===", flush=True)
    print(run_df[display_cols].to_string(index=False), flush=True)

    if RESULTS_CSV.exists():
        combined = pd.concat([pd.read_csv(RESULTS_CSV), run_df], ignore_index=True)
        combined.to_csv(RESULTS_CSV, index=False)
    else:
        run_df.to_csv(RESULTS_CSV, index=False)

    make_plot(run_df, ref_cont)
    print(f"\nUpdated {RESULTS_CSV} (+{len(run_df)} rows) and saved {PLOT_PNG}", flush=True)


if __name__ == "__main__":
    main()
