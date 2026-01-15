import pandas as pd
import numpy as np
import joblib

# ======================
# CONFIG
# ======================
DATA_FILE = "data/processed/model_dataset.csv"
MODEL_FILE = "models/xgb_tennis.pkl"
CALIBRATOR_FILE = "models/platt_calibrator.pkl"

START_BANKROLL = 1000.0
EV_THRESHOLD = 0.05
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.02

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_FILE)

# ======================
# LOAD MODEL
# ======================
model = joblib.load(MODEL_FILE)
calibrator = joblib.load(CALIBRATOR_FILE)

# ======================
# FEATURES
# ======================
FEATURES = [
    "elo_diff_surface",
    "elo_A_surface",
    "elo_B_surface",
    "last5_win_rate_diff",
    "h2h_win_pct_diff",
    "fatigue_diff",
    "log_odds_diff",
    "odds_A",
    "odds_B",
    "surface",
    "best_of",
    "tournament_level",
    "round_number",
]

X = df[FEATURES]
y = df["y"].values

# ======================
# MODEL PROBS
# ======================
p_raw = model.predict_proba(X)[:, 1]
p_model = calibrator.predict(p_raw.reshape(-1, 1))

df["p_model"] = p_model

# ======================
# POLYMARKET PROXY
# ======================
df["price_yes"] = 1.0 / df["odds_A"]

# ======================
# BACKTEST
# ======================
bankroll = START_BANKROLL
results = []

for _, row in df.iterrows():
    p = row["p_model"]
    price = row["price_yes"]
    outcome = row["y"]

    ev = p - price
    if ev < EV_THRESHOLD:
        continue

    kelly = ev / (1 - price)
    kelly = max(kelly, 0)

    stake = bankroll * KELLY_FRACTION * kelly
    stake = min(stake, bankroll * MAX_STAKE_PCT)

    if stake <= 0:
        continue

    bankroll -= stake

    if outcome == 1:
        payout = stake / price
        profit = payout - stake
        bankroll += payout
    else:
        profit = -stake

    results.append(profit)

# ======================
# RESULTS
# ======================
print("\n=== BACKTEST RESULTS ===")
print(f"Starting bankroll : ${START_BANKROLL:,.2f}")
print(f"Ending bankroll   : ${bankroll:,.2f}")
print(f"ROI               : {(bankroll / START_BANKROLL - 1) * 100:.2f}%")
print(f"Total bets        : {len(results)}")




