import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_utils import load_adult_train_test


ASSETS_DIR = Path("assets")
FIG_DIR = ASSETS_DIR / "figures"
TABLE_DIR = ASSETS_DIR / "tables"
MODEL_RESULTS = ASSETS_DIR / "model_results.json"
HYPERPARAM_SUMMARY = ASSETS_DIR / "hyperparameter_summary.csv"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_eda_artifacts(df: pd.DataFrame) -> None:
    # Missing values bar chart
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(x=missing.index, y=missing.values, color="#1f77b4")
        plt.title("Missing Values by Column")
        plt.ylabel("Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "missing_values_by_column.png", dpi=200)
        plt.close()

    # Target distribution bar chart
    target_counts = df["income"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    target_df = target_counts.rename_axis("income").reset_index(name="count")
    sns.barplot(data=target_df, x="income", y="count", hue="income", palette=["#2a9d8f", "#e76f51"], legend=False)
    plt.title("Income Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "income_class_distribution.png", dpi=200)
    plt.close()

    # Numeric histograms
    numeric_cols = ["age", "education_num", "hours_per_week"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=30, ax=axes[i], color="#264653")
        axes[i].set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "numeric_feature_histograms.png", dpi=200)
    plt.close(fig)


def save_model_tables() -> None:
    if HYPERPARAM_SUMMARY.exists():
        tuning_df = pd.read_csv(HYPERPARAM_SUMMARY)
        tuning_df = tuning_df.sort_values(by="best_cv_f1", ascending=False)
        tuning_df.to_csv(TABLE_DIR / "tuning_summary_ranked.csv", index=False)

    if MODEL_RESULTS.exists():
        payload = json.loads(MODEL_RESULTS.read_text(encoding="utf-8"))
        rows = []
        for model_name, metrics in payload.get("models", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "fit_seconds": metrics.get("fit_seconds"),
                }
            )
        if rows:
            model_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False)
            model_df.to_csv(TABLE_DIR / "model_metrics_ranked.csv", index=False)

            plt.figure(figsize=(8, 4))
            plot_df = model_df.melt(
                id_vars=["model"],
                value_vars=["accuracy", "precision", "recall", "f1"],
                var_name="metric",
                value_name="score",
            )
            sns.barplot(data=plot_df, x="metric", y="score", hue="model")
            plt.title("Model Comparison by Classification Metrics")
            plt.ylim(0.55, 0.90)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "model_comparison_metrics.png", dpi=200)
            plt.close()


def main() -> None:
    ensure_dirs()

    train_df, test_df = load_adult_train_test()
    full_df = pd.concat([train_df.assign(split="Train"), test_df.assign(split="Test")], ignore_index=True)

    save_eda_artifacts(full_df)
    save_model_tables()

    print("Saved report artifacts:")
    print(f"- Figures: {FIG_DIR}")
    print(f"- Tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
