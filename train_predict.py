
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "data" / "train-test.csv"
VALIDATION_FILE = PROJECT_DIR / "data" / "validation.csv"
DECEMBER_FILE = PROJECT_DIR / "data" / "december-chart-inputs.csv"

VALIDATION_OUTPUT = PROJECT_DIR / "validation_predictions.csv"
DECEMBER_OUTPUT = PROJECT_DIR / "december_chart_inputs_filled.csv"


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def add_date_features(df):
    """Create calendar features from the date column."""
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    df = df.drop(columns=["date"])

    return df


# ---------------------------------------------------------
# Load and clean training data
# ---------------------------------------------------------

print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop(columns=["posted_rate"]).copy()
y_train = train_df["posted_rate"].copy()

X_train["date"] = pd.to_datetime(X_train["date"])

# Negative weights are invalid.
# Treat them as missing so they can be imputed by the pipeline.
X_train.loc[X_train["weight"] < 0, "weight"] = np.nan

X_train = add_date_features(X_train)

# load_id is an identifier, not a predictive feature.
X_train_model = X_train.drop(columns=["load_id"])


# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

numeric_features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# ---------------------------------------------------------
# Final model
# ---------------------------------------------------------

gradient_boosting = Pipeline([
    ("preprocessor", preprocessor),
    (
        "to_dense",
        FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
    ),
    (
        "model",
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    ),
])

# Log-transform the target to reduce the influence of extreme
# high-rate observations while retaining them in training.
final_model = TransformedTargetRegressor(
    regressor=gradient_boosting,
    func=np.log1p,
    inverse_func=np.expm1,
)

print("Training final model...")
final_model.fit(X_train_model, y_train)

print("Final model trained successfully.")


# ---------------------------------------------------------
# Generate final validation predictions
# ---------------------------------------------------------

print("\nGenerating validation predictions...")

validation_df = pd.read_csv(VALIDATION_FILE)

validation_ids = validation_df["load_id"].copy()

X_validation = validation_df.drop(columns=["load_id"]).copy()

X_validation["date"] = pd.to_datetime(X_validation["date"])

X_validation.loc[
    X_validation["weight"] < 0,
    "weight"
] = np.nan

X_validation = add_date_features(X_validation)

validation_predictions = final_model.predict(X_validation)

validation_output = pd.DataFrame({
    "load_id": validation_ids,
    "predicted_rate": validation_predictions,
})

validation_output.to_csv(
    VALIDATION_OUTPUT,
    index=False
)

print(
    f"Saved {len(validation_output):,} validation predictions "
    f"to {VALIDATION_OUTPUT}"
)


# ---------------------------------------------------------
# Generate December predictions
# ---------------------------------------------------------

print("\nGenerating December predictions...")

december_df = pd.read_csv(DECEMBER_FILE)

december_input = december_df.drop(
    columns=["predicted_rate"]
).copy()

december_input["date"] = pd.to_datetime(
    december_input["date"]
)

# Recover the fixed coordinates for the cities from
# the training data.
pickup_coords = (
    train_df
    .groupby("pickup")[["pickup_lat", "pickup_lon"]]
    .first()
)

delivery_coords = (
    train_df
    .groupby("delivery")[["delivery_lat", "delivery_lon"]]
    .first()
)

december_input["pickup_lat"] = december_input["pickup"].map(
    pickup_coords["pickup_lat"]
)

december_input["pickup_lon"] = december_input["pickup"].map(
    pickup_coords["pickup_lon"]
)

december_input["delivery_lat"] = december_input["delivery"].map(
    delivery_coords["delivery_lat"]
)

december_input["delivery_lon"] = december_input["delivery"].map(
    delivery_coords["delivery_lon"]
)

# These variables are not provided in the December chart input.
# Leave them missing so the trained preprocessing pipeline
# applies the training-data median.
december_input["market_index"] = np.nan
december_input["quote_signal"] = np.nan

december_input = add_date_features(december_input)

december_predictions = final_model.predict(december_input)

december_output = december_df.copy()

december_output["predicted_rate"] = december_predictions

december_output.to_csv(
    DECEMBER_OUTPUT,
    index=False
)

print(
    f"Saved {len(december_output):,} December predictions "
    f"to {DECEMBER_OUTPUT}"
)

print("\nPrediction process completed successfully.")

