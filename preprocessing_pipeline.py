"""
preprocessing_pipeline.py
=========================
Reusable, leak-free preprocessing utilities for a tabular classification project.

Design rules (so the code stays leak-free):
  1. Anything that LEARNS from data (median, mean/std, category list, clipping
     bounds) lives inside a scikit-learn transformer, so it is learned in
     ``fit`` on the training set only and merely *applied* in ``transform``.
  2. Anything that is a fixed row-by-row rule (e.g. "FamilySize = SibSp + Parch + 1")
     is stateless, so it cannot leak information from one row to another.

Typical use
-----------
>>> from preprocessing_pipeline import *
>>> df = load_data()
>>> X, y = df.drop(columns=TARGET_COL), df[TARGET_COL]
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
>>> pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
...                       feature_engineer=TitanicFeatureEngineer())
>>> pipe.fit(X_train, y_train)          # preprocessing is fitted on X_train ONLY
>>> pipe.predict(X_test)               # X_test is only transformed, never fitted
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET_COL = "Survived"
TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
DEFAULT_LOCAL_PATH = Path(__file__).resolve().parent / "data" / "titanic.csv"


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(
    local_path: str | Path = DEFAULT_LOCAL_PATH,
    url: Optional[str] = TITANIC_URL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Load the dataset from a local CSV first, then fall back to a URL.

    Raises a clear error if neither source works or the target column is absent.
    """
    df: Optional[pd.DataFrame] = None
    local_path = Path(local_path)

    if local_path.exists():
        try:
            df = pd.read_csv(local_path)
            logger.info("Loaded %s from local file %s", df.shape, local_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s (%s); trying URL.", local_path, exc)

    if df is None and url:
        try:
            df = pd.read_csv(url)
            logger.info("Loaded %s from URL %s", df.shape, url)
        except Exception as exc:  # network errors come in many shapes
            raise RuntimeError(
                f"Could not load data from '{local_path}' or '{url}'. "
                "Place the CSV at the local path above and re-run."
            ) from exc

    if df is None:
        raise FileNotFoundError(f"No local file at {local_path} and no URL given.")
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {list(df.columns)}")
    return df


# --------------------------------------------------------------------------- #
# 2. Custom transformers
# --------------------------------------------------------------------------- #
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-wise feature engineering for the Titanic passenger table.

    Nothing is learned from the data, so there is no way to leak test information.
    Every rule is a fixed mapping decided in advance.

    New features:
    Title          : honorific extracted from Name, rare ones grouped (High-cardinality
                     Name -> 5 meaningful groups).
    FamilySize     : SibSp + Parch + 1 (the passenger plus relatives on board).
    IsAlone        : 1 if FamilySize == 1.
    Deck           : first letter of Cabin; 'U' (unknown) when Cabin is missing.
    FarePerPerson  : Fare / FamilySize (tickets are shared by groups).

    Dropped columns:
    PassengerId (row id), Name & Ticket (near-unique text), Cabin (replaced by Deck).
    Optionally SibSp / Parch (redundant with FamilySize) when
    ``drop_family_components=True``.
    """

    REQUIRED_COLUMNS = ("Name", "SibSp", "Parch", "Cabin", "Fare", "Ticket")
    TITLE_MAP = {
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    }
    KNOWN_DECKS = tuple("ABCDEFG")

    def __init__(self, drop_family_components: bool = False):
        self.drop_family_components = drop_family_components

    def _validate(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TitanicFeatureEngineer expects a pandas DataFrame.")
        missing = [c for c in self.REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required columns for feature engineering: {missing}")

    def fit(self, X, y=None):
        self._validate(X)  # nothing to learn: this transformer is stateless
        return self

    def transform(self, X):
        self._validate(X)
        out = X.copy()

        title = out["Name"].astype("string").str.extract(r",\s*([^\.]+)\.")[0].str.strip()
        out["Title"] = title.map(self.TITLE_MAP).fillna("Rare").astype(str)
        out.loc[out["Name"].isna(), "Title"] = "Unknown"

        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

        deck = out["Cabin"].astype("string").str[0]
        deck = deck.where(deck.isin(self.KNOWN_DECKS), "Other")
        out["Deck"] = deck.where(out["Cabin"].notna(), "U").astype(str)

        out["FarePerPerson"] = out["Fare"] / out["FamilySize"]

        to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
        if self.drop_family_components:
            to_drop += ["SibSp", "Parch"]
        return out.drop(columns=to_drop, errors="ignore")


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns: clip values outside [lower_q, upper_q] quantiles.

    The bounds are LEARNED in ``fit`` (training data only) and re-used in
    ``transform``, so extreme test values are clipped with the *training* bounds.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("Need 0 <= lower_quantile < upper_quantile <= 1.")
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.lower_bounds_ = np.nanquantile(arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


# --------------------------------------------------------------------------- #
# 3. Pipeline builders
# --------------------------------------------------------------------------- #
def build_preprocessor(
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> ColumnTransformer:
    """Build the ColumnTransformer (numeric branch + categorical branch).

    If the column lists are not supplied, columns are chosen by dtype *when the
    transformer is fitted* (numbers -> numeric branch, everything else -> categorical).
    """
    numeric_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(*clip_quantiles)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_sel = numeric_features if numeric_features is not None else make_column_selector(dtype_include=np.number)
    cat_sel = categorical_features if categorical_features is not None else make_column_selector(dtype_exclude=np.number)
    return ColumnTransformer(
        transformers=[("num", numeric_branch, num_sel), ("cat", categorical_branch, cat_sel)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, feature_engineer=None, preprocessor=None) -> Pipeline:
    """Full pipeline: [feature engineering] -> preprocessing -> estimator."""
    steps = []
    if feature_engineer is not None:
        steps.append(("features", feature_engineer))
    steps.append(("preprocess", preprocessor if preprocessor is not None else build_preprocessor()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# 4. Feature names, importance, evaluation helpers
# --------------------------------------------------------------------------- #
def get_feature_map(fitted_preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Map every transformed column back to the ORIGINAL feature it came from.

    Returns a DataFrame with columns: transformed_feature, original_feature, kind.
    Example row: ('cat__Sex_male', 'Sex', 'categorical').
    """
    rows_t, rows_o, rows_k = [], [], []
    for name, transformer, cols in fitted_preprocessor.transformers_:
        if transformer == "drop" or name == "remainder":
            continue
        cols = list(cols)
        out_names = list(transformer.get_feature_names_out(cols))
        last_step = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer
        if hasattr(last_step, "categories_"):  # one-hot: several columns per feature
            sizes = [len(c) for c in last_step.categories_]
            originals = list(np.repeat(cols, sizes))
            kind = "categorical"
        else:
            originals, kind = cols, "numeric"
        rows_t += [f"{name}__{n}" for n in out_names]
        rows_o += originals
        rows_k += [kind] * len(originals)

    fmap = pd.DataFrame({"transformed_feature": rows_t, "original_feature": rows_o, "kind": rows_k})
    expected = list(fitted_preprocessor.get_feature_names_out())
    if list(fmap["transformed_feature"]) != expected:
        raise RuntimeError("Feature map does not match ColumnTransformer output names.")
    return fmap


def compute_mutual_information(X_transformed, y, feature_map: pd.DataFrame) -> pd.DataFrame:
    """Mutual information between each transformed column and the target.

    One-hot columns are flagged as discrete; scaled numeric columns as continuous.
    Call this on TRAINING data only.
    """
    discrete_mask = (feature_map["kind"] == "categorical").to_numpy()
    mi = mutual_info_classif(
        X_transformed, y, discrete_features=discrete_mask, random_state=RANDOM_STATE
    )
    out = feature_map.assign(mutual_info=mi)
    return out.sort_values("mutual_info", ascending=False).reset_index(drop=True)


def aggregate_by_original(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Sum a per-column score over all one-hot columns of the same original feature."""
    return df.groupby("original_feature")[value_col].sum().sort_values(ascending=False)


def iqr_outlier_report(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Count values outside the Tukey fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n_out, "pct_outliers": 100 * n_out / len(s)})
    return pd.DataFrame(rows).set_index("feature")


def evaluate_classifier(fitted_pipeline: Pipeline, X, y_true) -> dict:
    """Return the standard binary-classification metrics for a fitted pipeline."""
    y_pred = fitted_pipeline.predict(X)
    y_proba = fitted_pipeline.predict_proba(X)[:, 1] if hasattr(fitted_pipeline, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}
