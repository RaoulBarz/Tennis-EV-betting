import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss

from xgboost import XGBClassifier

# ======================
# PATHS
# ======================
DATA_FILE = Path("data/processed/model_dataset.csv")
MODEL_OUT = Path("models/xgb_tennis.pkl")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_FILE)

TARGET = "y"

CATEGORICAL_FEATURES = [
    "surface",
    "tournament_level",
    "round_number"
]

NUMERIC_FEATURES = [
    "elo_diff_surface",
    "elo_A_surface",
    "elo_B_surface",
    "last5_win_rate_diff",
    "h2h_win_pct_diff",
    "fatigue_diff",
    "log_odds_diff",
    "odds_A",
    "odds_B",
    "best_of"
]


X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df[TARGET]

# ======================
# TIME-BASED SPLIT
# ======================
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

# ======================
# PREPROCESSING
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ]
)

# ======================
# MODEL
# ======================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=500,          # reduced since no early stopping
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", xgb)
    ]
)

# ======================
# TRAIN
# ======================
pipeline.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================
def evaluate(name, X_split, y_split):
    probs = pipeline.predict_proba(X_split)[:, 1]
    print(f"\n--- {name} ---")
    print("Log Loss :", log_loss(y_split, probs))
    print("Brier    :", brier_score_loss(y_split, probs))
    print("Mean P   :", probs.mean())
    print("Win Rate :", y_split.mean())

evaluate("TRAIN", X_train, y_train)
evaluate("VAL", X_val, y_val)
evaluate("TEST", X_test, y_test)

# ======================
# SAVE MODEL
# ======================
joblib.dump(pipeline, MODEL_OUT)
print(f"\n✔ Model saved → {MODEL_OUT}")
