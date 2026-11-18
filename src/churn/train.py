"""Train, tune, compare and persist churn models.

Usage
-----
    python -m churn.train            # full hyper-parameter search
    python -m churn.train --fast     # quick run (CI / smoke tests)

Methodology
-----------
1. Stratified 80/20 train/test split. The test set is touched only for final reporting.
2. Each candidate model is tuned with 5-fold stratified CV on the training set (ROC-AUC).
3. The winner is chosen by **cross-validated** ROC-AUC, not test-set score, so the
   reported test metrics are an unbiased estimate.
4. Models are trained without class re-weighting so that predicted probabilities stay
   calibrated (important for risk tiers and revenue-at-risk). Class imbalance is handled
   instead by tuning the decision threshold on out-of-fold predictions to maximise F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churn import __version__, config  # noqa: E402
from churn.data import load_dataset, split  # noqa: E402
from churn.explain import ChurnExplainer  # noqa: E402
from churn.features import FeatureEngineer, build_preprocessor  # noqa: E402
from churn.segment import segment_customers, segment_summary  # noqa: E402

logger = logging.getLogger("churn.train")

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def candidate_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    """Estimators and their hyper-parameter search spaces."""
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="lbfgs"),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=rs),
            {
                "model__max_depth": randint(3, 12),
                "model__min_samples_leaf": randint(5, 80),
                "model__min_samples_split": randint(2, 60),
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=1),
            {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": [6, 8, 10, 12, 16, None],
                "model__min_samples_leaf": randint(1, 20),
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
            },
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=rs,
                n_jobs=1,
            ),
            {
                "model__n_estimators": randint(150, 700),
                "model__max_depth": randint(2, 7),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 10),
                "model__reg_lambda": loguniform(0.1, 10),
                "model__gamma": uniform(0, 1),
            },
        ),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def capture_rate(y_true: np.ndarray, proba: np.ndarray, top_frac: float) -> float:
    """Share of all churners found in the top ``top_frac`` of customers by risk."""
    n = max(1, int(round(len(proba) * top_frac)))
    top = np.argsort(-proba)[:n]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
        "top20_capture": capture_rate(y_true, proba, 0.20),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
# Colour follows the model, never its rank.
MODEL_COLORS = dict(zip(MODEL_DISPLAY, PALETTE))


def _save(fig: plt.Figure, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(y_test: np.ndarray, probas: dict[str, np.ndarray]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, proba in probas.items():
        color = MODEL_COLORS[name]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc, ap = roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
        label = MODEL_DISPLAY[name]
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {auc:.3f})")
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{label} (AP {ap:.3f})")
        # Down-sample points for the dashboard JSON.
        idx_roc = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
        idx_pr = np.linspace(0, len(prec) - 1, min(200, len(prec))).astype(int)
        curves[name] = {
            "roc": {"fpr": fpr[idx_roc].round(4).tolist(), "tpr": tpr[idx_roc].round(4).tolist()},
            "pr": {
                "precision": prec[idx_pr].round(4).tolist(),
                "recall": rec[idx_pr].round(4).tolist(),
            },
        }
    ax_roc.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax_roc.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    base = y_test.mean()
    ax_pr.axhline(base, ls="--", color="#999", lw=1, label=f"Baseline ({base:.2f})")
    ax_pr.set(title="Precision-Recall curve (test set)", xlabel="Recall", ylabel="Precision")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", fontsize=9)
        ax.grid(alpha=0.3)
    _save(fig, "model_roc_pr_curves.png")
    return curves


def plot_comparison(comparison: pd.DataFrame) -> None:
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.2
    x = np.arange(len(metrics))
    ordered = comparison.set_index("key").loc[list(MODEL_DISPLAY)].reset_index()
    for i, row in ordered.iterrows():
        ax.bar(x + (i - 1.5) * width, row[metrics].astype(float), width * 0.92,
               color=MODEL_COLORS[row["key"]], label=row["model"])
    ax.set_xticks(x, [m.upper().replace("_", "-") for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the held-out test set (tuned threshold)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "model_comparison.png")


def plot_confusion(metrics: dict[str, float], model_label: str, threshold: float) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "#111")
    ax.set_xticks([0, 1], ["Stays", "Churns"])
    ax.set_yticks([0, 1], ["Stays", "Churns"])
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"{model_label}\nconfusion matrix @ threshold {threshold:.2f}")
    _save(fig, "confusion_matrix.png")


def plot_shap(explainer: ChurnExplainer, X_test: pd.DataFrame, importance: pd.DataFrame) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["label"], top["mean_abs_shap"], color=PALETTE[0])
    ax.set(title="Global feature importance (mean |SHAP|, test set)",
           xlabel=f"Mean |SHAP value| ({explainer.output_units})")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "shap_importance_bar.png")

    values = explainer.shap_values(X_test)
    Xt = explainer._transform(X_test)
    names = [n.split("__", 1)[-1] for n in explainer.transformed_names]
    shap.summary_plot(values, Xt, feature_names=names, max_display=18, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary (beeswarm) - test set", y=1.0)
    _save(fig, "shap_summary_beeswarm.png")


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train(fast: bool = False) -> dict[str, Any]:
    started = time.time()
    X, y, ids = load_dataset()
    X_train, X_test, y_train, y_test, ids_train, ids_test = split(X, y, ids)
    logger.info("Train %s | Test %s | churn rate %.3f", X_train.shape, X_test.shape, y.mean())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    n_iter = {"logistic_regression": 6, "decision_tree": 8, "random_forest": 4, "xgboost": 6} if fast else {
        "logistic_regression": 25, "decision_tree": 40, "random_forest": 30, "xgboost": 50
    }

    results, fitted, test_probas = [], {}, {}
    for name, (estimator, space) in candidate_models().items():
        t0 = time.time()
        search = RandomizedSearchCV(
            make_pipeline(estimator),
            space,
            n_iter=n_iter[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Out-of-fold probabilities on the training set -> unbiased threshold choice.
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_f1_threshold(y_train.to_numpy(), oof)

        proba = best.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), proba, threshold)
        metrics_default = evaluate(y_test.to_numpy(), proba, 0.5)

        row = {
            "key": name,
            "model": MODEL_DISPLAY[name],
            "cv_roc_auc": search.best_score_,
            "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
            "threshold": threshold,
            **metrics,
            "f1_at_0.5": metrics_default["f1"],
            "recall_at_0.5": metrics_default["recall"],
            "train_seconds": time.time() - t0,
            "best_params": json.dumps(
                {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                 for k, v in search.best_params_.items()}
            ),
        }
        results.append(row)
        fitted[name] = best
        test_probas[name] = proba
        logger.info(
            "%-20s CV AUC %.4f | test AUC %.4f | F1 %.4f @ %.2f | %.1fs",
            MODEL_DISPLAY[name], row["cv_roc_auc"], row["roc_auc"], row["f1"], threshold,
            row["train_seconds"],
        )

    comparison = pd.DataFrame(results).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best_key = comparison.loc[0, "key"]
    best_row = comparison.loc[0]
    best_pipeline = fitted[best_key]
    logger.info("Selected model: %s (highest CV ROC-AUC)", MODEL_DISPLAY[best_key])

    # ---- Persist comparison + curves -------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(config.COMPARISON_PATH, index=False)
    curves = plot_curves(y_test.to_numpy(), test_probas)
    config.CURVES_PATH.write_text(json.dumps(curves))
    plot_comparison(comparison)
    best_metrics = {k: best_row[k] for k in ["tn", "fp", "fn", "tp"]}
    plot_confusion(best_metrics, MODEL_DISPLAY[best_key], best_row["threshold"])

    # ---- Explainability ---------------------------------------------------
    explainer = ChurnExplainer(best_pipeline, X_train)
    importance = explainer.global_importance(X_test)
    importance.to_json(config.GLOBAL_IMPORTANCE_PATH, orient="records", indent=2)
    plot_shap(explainer, X_test, importance)

    # ---- Model bundle -----------------------------------------------------
    high_value_threshold = float(X_train["MonthlyCharges"].median())
    background = X_train.sample(200, random_state=config.RANDOM_STATE)
    bundle = {
        "pipeline": best_pipeline,
        "background": background,
        "threshold": float(best_row["threshold"]),
        "high_value_threshold": high_value_threshold,
        "model_key": best_key,
        "model_name": MODEL_DISPLAY[best_key],
        "version": __version__,
    }
    joblib.dump(bundle, config.MODEL_PATH, compress=3)

    metric_keys = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier", "top20_capture"]
    metadata = {
        "model_name": MODEL_DISPLAY[best_key],
        "model_key": best_key,
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selection_metric": "cv_roc_auc",
        "cv_roc_auc": round(float(best_row["cv_roc_auc"]), 4),
        "decision_threshold": round(float(best_row["threshold"]), 4),
        "high_value_threshold": high_value_threshold,
        "test_metrics": {k: round(float(best_row[k]), 4) for k in metric_keys},
        "best_params": json.loads(best_row["best_params"]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "shap_output_units": explainer.output_units,
        "risk_tiers": config.RISK_TIERS,
        "features": config.RAW_FEATURES,
        "fast_mode": fast,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    # ---- Score the whole customer base for the dashboard -------------------
    proba_all = best_pipeline.predict_proba(X)[:, 1]
    scored = segment_customers(X.reset_index(drop=True), proba_all, high_value_threshold)
    scored.insert(0, config.ID_COL, ids.to_numpy())
    scored["actual_churn"] = y.to_numpy()
    scored["split"] = np.where(ids.isin(set(ids_test)), "test", "train")
    scored.to_csv(config.SCORED_PATH, index=False)
    segment_summary(scored).to_csv(config.PROCESSED_DIR / "segment_summary.csv", index=False)

    logger.info("Done in %.1fs. Artifacts written to %s", time.time() - started, config.MODELS_DIR)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--fast", action="store_true", help="small search for CI / smoke tests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = train(fast=args.fast)
    print(json.dumps(metadata["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
