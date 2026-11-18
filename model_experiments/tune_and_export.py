"""Bayesian hyperparameter tuning (Optuna TPE) and final LightGBM export for PathSnap."""

from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

# Paths relative to PathSnap repository root
TRAIN_CSV = Path("full_features/train_candidate_features_sample.csv")
TEST_CSV = Path("full_features/student_test_candidate_features_sample.csv")
TEST_DATA_DIR = Path("data/student_test")
RESULTS_CSV = Path("model_experiments/results.csv")
TRIALS_CSV = Path("model_experiments/optuna_trials.csv")
COMPARISON_CSV = Path("model_experiments/tuned_vs_untuned.csv")
ARTIFACTS_DIR = Path("model_artifacts")

SEED = 42
CANDIDATE_LIMIT = 8
N_SPLITS = 5
POSITIVE_CLASS = 1
OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT_S = 900  # 15 min cap (strictly within 30 trials / 20 min budget)

# Exact 13 clean geometry/motion features verified from model_experiments/harness.py
FEATURE_COLUMNS = [
    "timestamp_gap_s",
    "step_distance_m",
    "point_speed_mps",
    "heading_deg",
    "turn_angle_deg",
    "perpendicular_distance_m",
    "along_arc_fraction",
    "arc_length_m",
    "arc_bearing_deg",
    "heading_delta_deg",
    "arc_out_degree",
    "arc_in_degree",
    "candidate_rank",
]
EXCLUDED_COLUMNS = [
    "trip_id", "point_idx", "arc_id", "longitude", "latitude", "timestamp", "label"
]

UNTUNED_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def build_pipeline(params: dict) -> Pipeline:
    """Wrap LightGBM in an sklearn Pipeline with median imputation for safe API inference."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(**params)),
    ])


def top1_accuracy(df: pd.DataFrame, probs: np.ndarray) -> float:
    """Per (trip_id, point_idx), select highest-prob candidate and check label == 1."""
    tmp = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "point_idx": df["point_idx"].to_numpy(),
        "label": df["label"].to_numpy(),
        "prob": probs,
    })
    best_idx = tmp.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    return float(tmp.loc[best_idx, "label"].mean())


def load_trip_arcs(trip_ids, data_dir: Path) -> dict[int, np.ndarray]:
    """Load each trip's .arcs file as (N, 2) array of [from_node, to_node]."""
    arcs = {}
    for tid in trip_ids:
        tid_str = f"{int(tid):08d}"
        path = data_dir / tid_str / f"{tid_str}.arcs"
        arcs[int(tid)] = np.atleast_2d(np.loadtxt(path, dtype=int))
    return arcs


def route_continuity(df: pd.DataFrame, probs: np.ndarray, arcs_by_trip: dict) -> float:
    """Mean per-trip fraction of consecutive unique predicted arcs connected in .arcs."""
    tmp = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "point_idx": df["point_idx"].to_numpy(),
        "arc_id": df["arc_id"].to_numpy(),
        "prob": probs,
    })
    best = tmp.loc[tmp.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()]
    scores = []
    for tid, grp in best.groupby("trip_id"):
        seq = grp.sort_values("point_idx")["arc_id"].to_numpy()
        collapsed = [int(seq[0])] + [
            int(a) for a, prev in zip(seq[1:], seq[:-1]) if a != prev
        ]
        if len(collapsed) < 2:
            scores.append(1.0)
            continue
        nodes = arcs_by_trip[int(tid)]
        connected = sum(
            nodes[a1, 1] == nodes[a2, 0]
            for a1, a2 in zip(collapsed[:-1], collapsed[1:])
            if 0 <= a1 < len(nodes) and 0 <= a2 < len(nodes)
        )
        scores.append(connected / (len(collapsed) - 1))
    return float(np.mean(scores))


def extract_pos_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probability using model.classes_ explicitly."""
    classes = list(model.classes_)
    assert POSITIVE_CLASS in classes, f"Positive class {POSITIVE_CLASS} missing from {classes}"
    pos_idx = classes.index(POSITIVE_CLASS)
    return model.predict_proba(X)[:, pos_idx]


def cv_evaluate(params: dict, train_df: pd.DataFrame, splits: list) -> dict:
    """Evaluate a LightGBM config via GroupKFold on the 80 training trips only."""
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"].to_numpy()
    f1s, top1s, lls = [], [], []
    t0 = time.perf_counter()
    for tr_idx, va_idx in splits:
        pipe = build_pipeline(params)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        va_prob = extract_pos_proba(pipe, X.iloc[va_idx])
        va_pred = (va_prob >= 0.5).astype(int)
        f1s.append(f1_score(y[va_idx], va_pred, average="macro"))
        top1s.append(top1_accuracy(train_df.iloc[va_idx], va_prob))
        lls.append(log_loss(y[va_idx], va_prob))
    return {
        "cv_macro_f1_mean": float(np.mean(f1s)),
        "cv_macro_f1_std": float(np.std(f1s)),
        "cv_top1_mean": float(np.mean(top1s)),
        "cv_top1_std": float(np.std(top1s)),
        "cv_log_loss_mean": float(np.mean(lls)),
        "cv_time_s": float(time.perf_counter() - t0),
    }


def fit_and_report_student_test(
    params: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    arcs_by_trip: dict,
) -> tuple[Pipeline, dict]:
    """Fit on all 80 training trips; report evaluation-only metrics on 10 student-test trips."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].to_numpy()
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].to_numpy()

    pipe = build_pipeline(params)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    tr_prob = extract_pos_proba(pipe, X_train)
    tr_pred = (tr_prob >= 0.5).astype(int)

    t0 = time.perf_counter()
    te_prob = extract_pos_proba(pipe, X_test)
    infer_time = time.perf_counter() - t0
    assert np.all(np.isfinite(te_prob)) and np.all((te_prob >= 0.0) & (te_prob <= 1.0))
    te_pred = (te_prob >= 0.5).astype(int)

    metrics = {
        "train_macro_f1": round(float(f1_score(y_train, tr_pred, average="macro")), 4),
        "train_top1_acc": round(top1_accuracy(train_df, tr_prob), 4),
        "test_macro_f1": round(float(f1_score(y_test, te_pred, average="macro")), 4),
        "test_precision": round(float(precision_score(y_test, te_pred, average="macro", zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, te_pred, average="macro", zero_division=0)), 4),
        "test_log_loss": round(float(log_loss(y_test, te_prob)), 4),
        "test_top1_acc": round(top1_accuracy(test_df, te_prob), 4),
        "test_route_continuity": round(route_continuity(test_df, te_prob, arcs_by_trip), 4),
        "train_time_s": round(train_time, 3),
        "infer_time_s": round(infer_time, 3),
    }
    return pipe, metrics


def main() -> None:
    print("Loading full training and student-test feature tables...", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Schema and data integrity assertions
    assert set(FEATURE_COLUMNS).issubset(train_df.columns), "Missing train feature columns"
    assert set(FEATURE_COLUMNS).issubset(test_df.columns), "Missing test feature columns"
    assert set(EXCLUDED_COLUMNS).isdisjoint(FEATURE_COLUMNS), "Excluded col leaked into features"
    assert train_df["trip_id"].nunique() == 80, f"Expected 80 train trips, got {train_df['trip_id'].nunique()}"
    assert test_df["trip_id"].nunique() == 10, f"Expected 10 test trips, got {test_df['trip_id'].nunique()}"
    assert set(train_df["label"].unique()) == {0, 1}, f"Unexpected train labels: {train_df['label'].unique()}"

    # Use float32 for memory/speed consistency across CV and final fit
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].astype(np.float32)
    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].astype(np.float32)

    print(f"Train: {train_df.shape} ({train_df['trip_id'].nunique()} trips) | "
          f"Student-test: {test_df.shape} ({test_df['trip_id'].nunique()} trips)", flush=True)
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}\n", flush=True)

    # Precompute deterministic 5-fold GroupKFold splits on the 80 training trips only
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(train_df[FEATURE_COLUMNS], train_df["label"].to_numpy(), train_df["trip_id"].to_numpy()))

    # 1) Untuned baseline CV on 80 training trips
    print("[Untuned LightGBM Baseline] Running 5-fold GroupKFold CV...", flush=True)
    untuned_cv = cv_evaluate(UNTUNED_PARAMS, train_df, splits)
    print(
        f"  Untuned CV: F1={untuned_cv['cv_macro_f1_mean']:.4f} +/- {untuned_cv['cv_macro_f1_std']:.4f}, "
        f"Top-1={untuned_cv['cv_top1_mean']:.4f}, LogLoss={untuned_cv['cv_log_loss_mean']:.4f} "
        f"({untuned_cv['cv_time_s']:.1f}s)\n",
        flush=True,
    )

    # 2) Optuna TPE Bayesian optimisation on 80 training trips ONLY
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_lgbm_tpe")

    # Enqueue baseline-like starting point so TPE starts from a strong anchor
    study.enqueue_trial({
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
    })

    trial_records = []
    t_opt_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        res = cv_evaluate(params, train_df, splits)
        trial.set_user_attr("cv_macro_f1_std", res["cv_macro_f1_std"])
        trial.set_user_attr("cv_top1_mean", res["cv_top1_mean"])
        trial.set_user_attr("cv_top1_std", res["cv_top1_std"])
        trial.set_user_attr("cv_log_loss_mean", res["cv_log_loss_mean"])
        trial.set_user_attr("cv_time_s", res["cv_time_s"])

        rec = {
            "trial": trial.number,
            "cv_macro_f1_mean": round(res["cv_macro_f1_mean"], 4),
            "cv_macro_f1_std": round(res["cv_macro_f1_std"], 4),
            "cv_top1_mean": round(res["cv_top1_mean"], 4),
            "cv_top1_std": round(res["cv_top1_std"], 4),
            "cv_log_loss_mean": round(res["cv_log_loss_mean"], 4),
            "cv_time_s": round(res["cv_time_s"], 2),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in trial.params.items()},
        }
        trial_records.append(rec)
        print(
            f"  Trial {trial.number + 1:02d}/{OPTUNA_TRIALS}: "
            f"CV F1={res['cv_macro_f1_mean']:.4f} +/- {res['cv_macro_f1_std']:.4f} | "
            f"Top-1={res['cv_top1_mean']:.4f} | LogLoss={res['cv_log_loss_mean']:.4f} "
            f"({res['cv_time_s']:.1f}s)",
            flush=True,
        )
        return res["cv_macro_f1_mean"]

    print(f"[Optuna TPE] Starting up to {OPTUNA_TRIALS} trials (timeout={OPTUNA_TIMEOUT_S}s, seed={SEED})...", flush=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_S)
    optuna_total_time = time.perf_counter() - t_opt_start

    trials_df = pd.DataFrame(trial_records)
    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(TRIALS_CSV, index=False)
    print(f"\nSaved {len(trials_df)} Optuna trials to {TRIALS_CSV} (total {optuna_total_time:.1f}s)", flush=True)

    best_trial = study.best_trial
    best_optuna_params = {
        **best_trial.params,
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    best_optuna_cv = {
        "cv_macro_f1_mean": float(best_trial.value),
        "cv_macro_f1_std": float(best_trial.user_attrs["cv_macro_f1_std"]),
        "cv_top1_mean": float(best_trial.user_attrs["cv_top1_mean"]),
        "cv_top1_std": float(best_trial.user_attrs["cv_top1_std"]),
        "cv_log_loss_mean": float(best_trial.user_attrs["cv_log_loss_mean"]),
    }

    # Select final config strictly by 80-trip GroupKFold CV Macro F1
    if best_optuna_cv["cv_macro_f1_mean"] >= untuned_cv["cv_macro_f1_mean"]:
        selected_label = "LightGBM (Optuna Tuned)"
        final_params = best_optuna_params
        final_cv = best_optuna_cv
    else:
        selected_label = "LightGBM (Untuned Baseline)"
        final_params = UNTUNED_PARAMS
        final_cv = untuned_cv

    print(f"\nSelected by 80-trip CV Macro F1: {selected_label} (CV F1={final_cv['cv_macro_f1_mean']:.4f})", flush=True)

    # 3) Fit untuned and tuned on all 80 training trips; report student-test metrics (eval-only)
    arcs_by_trip = load_trip_arcs(sorted(test_df["trip_id"].unique()), TEST_DATA_DIR)
    print("\nFitting untuned and tuned models on all 80 training trips...", flush=True)
    untuned_pipe, untuned_test = fit_and_report_student_test(UNTUNED_PARAMS, train_df, test_df, arcs_by_trip)
    tuned_pipe, tuned_test = fit_and_report_student_test(best_optuna_params, train_df, test_df, arcs_by_trip)

    final_pipe = tuned_pipe if final_params is best_optuna_params else untuned_pipe
    final_test = tuned_test if final_params is best_optuna_params else untuned_test

    # Save tuned vs untuned comparison CSV
    comp_df = pd.DataFrame([
        {
            "variant": "Untuned LightGBM",
            "cv_macro_f1_mean": round(untuned_cv["cv_macro_f1_mean"], 4),
            "cv_macro_f1_std": round(untuned_cv["cv_macro_f1_std"], 4),
            "cv_top1_mean": round(untuned_cv["cv_top1_mean"], 4),
            "cv_top1_std": round(untuned_cv["cv_top1_std"], 4),
            "cv_log_loss_mean": round(untuned_cv["cv_log_loss_mean"], 4),
            **untuned_test,
        },
        {
            "variant": "Optuna Tuned LightGBM",
            "cv_macro_f1_mean": round(best_optuna_cv["cv_macro_f1_mean"], 4),
            "cv_macro_f1_std": round(best_optuna_cv["cv_macro_f1_std"], 4),
            "cv_top1_mean": round(best_optuna_cv["cv_top1_mean"], 4),
            "cv_top1_std": round(best_optuna_cv["cv_top1_std"], 4),
            "cv_log_loss_mean": round(best_optuna_cv["cv_log_loss_mean"], 4),
            **tuned_test,
        },
    ])
    comp_df.to_csv(COMPARISON_CSV, index=False)
    print("\n=== UNTUNED VS TUNED COMPARISON ===", flush=True)
    print(comp_df.to_string(index=False), flush=True)

    # Append final tuned row to model_experiments/results.csv
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg_summary = (
        f"Optuna TPE ({len(trials_df)} trials): n_est={final_params['n_estimators']}, "
        f"lr={final_params['learning_rate']:.4f}, leaves={final_params['num_leaves']}, "
        f"depth={final_params['max_depth']}, min_child={final_params['min_child_samples']}"
    )
    new_row = pd.DataFrame([{
        "timestamp": run_ts,
        "train_file": "full_features/train_candidate_features_sample.csv",
        "feature_set": "clean_13",
        "n_features": len(FEATURE_COLUMNS),
        "model": selected_label,
        "family": "Boosting",
        "config": cfg_summary,
        "seed": SEED,
        "cv_macro_f1_mean": round(final_cv["cv_macro_f1_mean"], 4),
        "cv_macro_f1_std": round(final_cv["cv_macro_f1_std"], 4),
        "cv_top1_mean": round(final_cv["cv_top1_mean"], 4),
        "cv_top1_std": round(final_cv["cv_top1_std"], 4),
        "cv_log_loss_mean": round(final_cv["cv_log_loss_mean"], 4),
        **final_test,
    }])
    if RESULTS_CSV.exists():
        pd.concat([pd.read_csv(RESULTS_CSV), new_row], ignore_index=True).to_csv(RESULTS_CSV, index=False)
    else:
        new_row.to_csv(RESULTS_CSV, index=False)

    # 4) Export model artifacts for FastAPI backend
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "final_model.joblib"
    feat_path = ARTIFACTS_DIR / "feature_columns.json"
    cfg_path = ARTIFACTS_DIR / "config.json"
    req_path = ARTIFACTS_DIR / "requirements-model.txt"

    joblib.dump(final_pipe, model_path, compress=3)
    feat_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2) + "\n", encoding="utf-8")

    # Collect exact runtime package versions from the active .venv
    pkg_names = ["joblib", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy"]
    pkg_versions = {p: importlib.metadata.version(p) for p in pkg_names}
    pkg_versions_all = {**pkg_versions, "optuna": importlib.metadata.version("optuna")}

    req_lines = [f"{p}=={pkg_versions[p]}" for p in pkg_names]
    req_path.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    classes_list = [int(c) for c in final_pipe.classes_]
    pos_class_index = classes_list.index(POSITIVE_CLASS)

    config_payload = {
        "model_name": selected_label,
        "algorithm_family": "Boosting (LightGBM)",
        "pipeline_steps": ["SimpleImputer(strategy='median')", "LGBMClassifier"],
        "hyperparameters": final_params,
        "random_seed": SEED,
        "candidate_limit": CANDIDATE_LIMIT,
        "positive_class_label": POSITIVE_CLASS,
        "classes": classes_list,
        "positive_class_index": pos_class_index,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "training_configuration": {
            "dataset_path": str(TRAIN_CSV).replace("\\", "/"),
            "num_training_trips": int(train_df["trip_id"].nunique()),
            "num_training_rows": int(len(train_df)),
            "positive_label_count": int((train_df["label"] == POSITIVE_CLASS).sum()),
            "negative_label_count": int((train_df["label"] != POSITIVE_CLASS).sum()),
            "positive_rate": round(float((train_df["label"] == POSITIVE_CLASS).mean()), 6),
            "feature_dtype": "float32",
            "train_time_s": final_test["train_time_s"],
            "trained_at": run_ts,
        },
        "group_kfold_validation": {
            "strategy": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_column": "trip_id",
            "selection_metric": "cv_macro_f1_mean",
            "untuned_baseline": {k: round(v, 4) for k, v in untuned_cv.items()},
            "selected_final": {k: round(v, 4) for k, v in final_cv.items() if k != "cv_time_s"},
        },
        "optuna_tuning": {
            "sampler": "TPESampler",
            "seed": SEED,
            "objective": "maximize 5-fold GroupKFold CV macro F1 on 80 training trips",
            "max_trials": OPTUNA_TRIALS,
            "timeout_s": OPTUNA_TIMEOUT_S,
            "completed_trials": int(len(trials_df)),
            "best_trial_number": int(best_trial.number),
            "total_tuning_time_s": round(optuna_total_time, 2),
            "trials_log_path": str(TRIALS_CSV).replace("\\", "/"),
        },
        "student_test_evaluation_only": {
            "note": "Reported for final comparison only; never used for fitting, tuning, or selection.",
            "num_student_test_trips": int(test_df["trip_id"].nunique()),
            "num_student_test_rows": int(len(test_df)),
            "untuned": untuned_test,
            "selected_final": final_test,
        },
        "environment": {
            "python_version": platform.python_version(),
            "packages": pkg_versions_all,
        },
        "feature_generator_reference": {
            "script": "data_exploration/code/generate_candidate_features.py",
            "function": "generate_feature_rows(trip_dir, candidate_limit=8, include_label=False)",
            "cli_command": (
                "python data_exploration/code/generate_candidate_features.py "
                "--data-dir data --output-dir full_features --candidate-limit 8 --sample-per-partition 80"
            ),
        },
    }
    cfg_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== EXPORTED ARTIFACTS ===", flush=True)
    for p in (model_path, feat_path, cfg_path, req_path):
        print(f"  {p} ({p.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
