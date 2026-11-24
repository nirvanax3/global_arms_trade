"""
Machine Learning models for the Global Arms Trade dashboard.

Model A - Delivery Delay Predictor (supervised REGRESSION)
    Predicts how many years an arms deal takes from order to delivery,
    using: exporter, importer, weapon category, quantity, order year and
    whether the order year was estimated.
    Models compared: Baseline (predict the mean) vs Linear Regression vs Random Forest.

Model B - Weapon Category Classifier (supervised CLASSIFICATION, NLP)
    Predicts the weapon category (Aircraft, Missiles, Ships, ...) from the
    weapon's text: its description + designation, e.g. "portable SAM Strela-2".
    Text is converted to numbers with TF-IDF.
    Models compared: Multinomial Naive Bayes vs Logistic Regression.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42  # fixed seed so results are reproducible
TEST_SIZE = 0.2    # 80% training data, 20% test data

# Orders placed after this year are left out of training: many of their
# long-delay deliveries have not happened yet (data ends in 2023), so they
# would wrongly teach the model that recent orders are always delivered fast.
# This problem is called "right-censoring".
DELAY_TRAIN_MAX_YEAR = 2018

CATEGORICAL_FEATURES = ["source", "target", "category"]
NUMERIC_FEATURES = ["log_quantity", "order_year", "order_year_estimated"]
DELAY_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# =====================================================================
# MODEL A: Delivery Delay Predictor (Regression)
# =====================================================================
def prepare_delay_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling table: one row per deal, features + target (delay_years)."""
    data = df.copy()
    order = pd.to_datetime(data["order_date"], errors="coerce")
    delivery = pd.to_datetime(data["delivery_date"], errors="coerce")

    # Target variable: years between order and delivery
    data["delay_years"] = (delivery - order).dt.days / 365.25
    data["order_year"] = order.dt.year

    # Quantity is very skewed (median 10, max 25,000) -> log transform
    data["log_quantity"] = np.log1p(data["quantity"])
    data["order_year_estimated"] = data["order_year_estimated"].astype(int)

    data = data.dropna(subset=["delay_years", "quantity", "order_year"])
    data = data[data["delay_years"] >= 0]                       # remove impossible values
    data = data[data["order_year"] <= DELAY_TRAIN_MAX_YEAR]     # avoid censoring bias
    return data


def make_delay_preprocessor() -> ColumnTransformer:
    """One-hot encode country/category names; pass numeric columns through unchanged.
    Countries with fewer than 10 deals are grouped into one 'infrequent' column,
    and countries never seen in training are handled without crashing."""
    return ColumnTransformer(
        [("cat",
          OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
          CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )


def train_delay_models(df: pd.DataFrame) -> dict:
    data = prepare_delay_data(df)
    X = data[DELAY_FEATURES]
    y = data["delay_years"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    candidates = {
        "Baseline (always predicts average)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", make_delay_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "MAE (years)": mean_absolute_error(y_test, pred),
            "RMSE (years)": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        })
        fitted[name] = pipe

    best = fitted["Random Forest"]

    # Permutation importance: shuffle one feature at a time and measure how
    # much worse the model gets. Bigger drop = more important feature.
    perm = permutation_importance(
        best, X_test, y_test, n_repeats=5,
        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error",
    )
    importance = (
        pd.DataFrame({"Feature": DELAY_FEATURES, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": best,
        "metrics": pd.DataFrame(rows),
        "importance": importance,
        "y_test": y_test.to_numpy(),
        "y_pred": best.predict(X_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sources": sorted(data["source"].unique()),
        "targets": sorted(data["target"].unique()),
        "categories": sorted(data["category"].unique()),
    }


def predict_delay(model: Pipeline, source, target, category, quantity,
                  order_year, order_year_estimated=False) -> dict:
    """Predict delay for one new deal. Also returns a likely range using the
    spread of predictions from the individual trees in the Random Forest."""
    row = pd.DataFrame([{
        "source": source,
        "target": target,
        "category": category,
        "log_quantity": np.log1p(quantity),
        "order_year": order_year,
        "order_year_estimated": int(order_year_estimated),
    }])
    prediction = float(model.predict(row)[0])

    forest = model.named_steps["model"]
    row_encoded = model.named_steps["preprocess"].transform(row)
    tree_preds = np.array([tree.predict(row_encoded)[0] for tree in forest.estimators_])
    low, high = np.percentile(tree_preds, [10, 90])

    return {"prediction": prediction, "low": float(low), "high": float(high)}


# =====================================================================
# MODEL B: Weapon Category Classifier (NLP / Text Classification)
# =====================================================================
def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["description", "category"]).copy()
    data["text"] = (
        data["description"].fillna("") + " " + data["designation"].fillna("")
    ).str.lower().str.strip()

    # IMPORTANT: the dataset repeats the same weapon many times (only 193
    # unique descriptions in 10,520 rows). Without removing duplicates, the
    # same text would appear in both train and test, so the model would just
    # memorise it and show a fake 100% accuracy (data leakage).
    return data[["text", "category"]].drop_duplicates().reset_index(drop=True)


def train_category_models(df: pd.DataFrame) -> dict:
    data = prepare_text_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["category"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=data["category"],
    )

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        pipe = Pipeline([
            # TF-IDF turns text into numbers: words that are frequent in one
            # description but rare overall get a high weight.
            # ngram_range=(1, 2) uses single words AND word pairs ("transport aircraft").
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
        })
        fitted[name] = pipe

    best = fitted["Logistic Regression"]
    y_pred = best.predict(X_test)
    labels = list(best.classes_)

    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T

    return {
        "model": best,
        "metrics": pd.DataFrame(rows),
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels
        ),
        "report": report,
        "n_unique": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_category(model: Pipeline, text: str, top_n: int = 3) -> pd.DataFrame:
    probs = model.predict_proba([text.lower().strip()])[0]
    return (
        pd.DataFrame({"Category": model.classes_, "Probability": probs})
        .sort_values("Probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
