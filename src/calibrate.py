import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

# ======================
# PATHS
# ======================
DATA_FILE = Path("data/processed/model_dataset.csv")
MODEL_FILE = Path("models/xgb_tennis.pkl")
CALIBRATOR_OUT = Path("models/platt_calibrator.pkl")

# ======================
# LOAD DATA + MODEL
# ======================
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

TARGET = "y"

FEATURES = [
    "surface",
    "tournament_level",
    "round_number",
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

X = df[FEATURES]
y = df[TARGET]

# ======================
# SAME TIME-BASED SPLIT
# ======================
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val   = X.iloc[train_end:val_end]
y_val   = y.iloc[train_end:val_end]

X_test  = X.iloc[val_end:]
y_test  = y.iloc[val_end:]

# ======================
# RAW PROBABILITIES
# ======================
p_val_raw = model.predict_proba(X_val)[:, 1]
p_test_raw = model.predict_proba(X_test)[:, 1]

# ======================
# FIT PLATT CALIBRATOR
# ======================
calibrator = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

# Important: reshape to 2D
calibrator.fit(p_val_raw.reshape(-1, 1), y_val)

# ======================
# CALIBRATED PROBABILITIES
# ======================
p_val_cal = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]
p_test_cal = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

# ======================
# METRICS COMPARISON
# ======================
def report(name, y_true, p_raw, p_cal):
    print(f"\n--- {name} ---")
    print("RAW Log Loss :", log_loss(y_true, p_raw))
    print("CAL Log Loss :", log_loss(y_true, p_cal))
    print("RAW Brier    :", brier_score_loss(y_true, p_raw))
    print("CAL Brier    :", brier_score_loss(y_true, p_cal))
    print("RAW Mean P   :", p_raw.mean())
    print("CAL Mean P   :", p_cal.mean())
    print("Win Rate     :", y_true.mean())

report("VALIDATION", y_val, p_val_raw, p_val_cal)
report("TEST", y_test, p_test_raw, p_test_cal)

# ======================
# SAVE CALIBRATOR
# ======================
joblib.dump(calibrator, CALIBRATOR_OUT)
print(f"\n✔ Calibrator saved → {CALIBRATOR_OUT}")
