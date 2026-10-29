"""Train and compare churn models with cross-validated hyperparameter tuning."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"

TARGET = "churn_value"
# cltv is IBM's calculated value; we keep it out of the features and use it later
# only in the business / cost-benefit step.
EXCLUDED_FEATURES = ["cltv"]
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    """Return a stratified 80/20 train/test split (same split every run)."""
    X = df.drop(columns=[TARGET] + EXCLUDED_FEATURES)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )


def get_models_and_grids(scale_pos_weight: float) -> dict:
    """Return each model with its hyperparameter grid (keys use the 'clf__' prefix)."""
    return {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "clf__max_depth": [3, 5, 7, 10],
                "clf__min_samples_leaf": [10, 30, 50],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [6, 10, None],
                "clf__min_samples_leaf": [5, 20],
            },
        ),
        "xgboost": (
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    """Compute test metrics at the default 0.5 threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def save_feature_importance(name: str, pipeline: Pipeline, top_n: int = 15) -> None:
    """Plot top features using coefficients (linear) or importances (tree models)."""
    clf = pipeline.named_steps["clf"]
    names = pipeline.named_steps["prep"].get_feature_names_out()

    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
        xlabel = "Coefficient (positive = more churn)"
    else:
        values = clf.feature_importances_
        xlabel = "Importance"

    importance = pd.Series(values, index=names)
    top = importance.reindex(importance.abs().sort_values(ascending=False).index).head(top_n)

    print(f"\nTop features for {name}:")
    print(top.round(3).to_string())

    ax = sns.barplot(x=top.values, y=top.index, orient="h")
    ax.set_title(f"Top {top_n} features: {name}")
    ax.set_xlabel(xlabel)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"20_feature_importance_{name}.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

    # Ratio of negative to positive samples, used by XGBoost to handle imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results = []
    for name, (estimator, grid) in get_models_and_grids(scale_pos_weight).items():
        print(f"\n=== Tuning {name} ===")
        pipeline = Pipeline(
            [("prep", build_preprocessor(X_train)), ("clf", estimator)]
        )
        search = GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)

        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics.update(
            {
                "model": name,
                "cv_roc_auc": search.best_score_,
                "best_params": search.best_params_,
            }
        )
        results.append(metrics)
        print(f"Best params: {search.best_params_}")
        print(f"CV ROC-AUC: {search.best_score_:.4f} | Test ROC-AUC: {metrics['roc_auc']:.4f}")

        joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}.joblib")
        save_feature_importance(name, search.best_estimator_)

    table = (
        pd.DataFrame(results)
        .set_index("model")[["cv_roc_auc", "roc_auc", "precision", "recall", "f1", "best_params"]]
        .sort_values("roc_auc", ascending=False)
    )
    table.to_csv(REPORT_DIR / "model_comparison.csv")

    print("\n=== Model comparison (test set, threshold = 0.5) ===")
    print(table.drop(columns="best_params").round(3).to_string())
    print(f"\nSaved comparison to {REPORT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()