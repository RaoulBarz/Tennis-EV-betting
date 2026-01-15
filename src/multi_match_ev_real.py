import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from bisect import bisect_left
from collections import defaultdict

# ======================
# LOAD MODELS
# ======================
model = joblib.load("models/xgb_tennis.pkl")
calibrator = joblib.load("models/platt_calibrator.pkl")

# ======================
# LOAD DATA
# ======================
MATCHES_FILE = Path("data/processed/atp_matches_all.csv")
ELO_FILE = Path("data/external/ta_atp_elo_surfaces.csv")

df = pd.read_csv(MATCHES_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

elo_df = pd.read_csv(ELO_FILE)
elo_df["player"] = elo_df["player"].astype(str).str.strip()
elo_df = elo_df.set_index("player")

# ======================
# PRECOMPUTE PLAYER HISTORY (FAST)
# ======================
player_dates = defaultdict(list)
player_results = defaultdict(list)
player_opponents = defaultdict(list)

for _, r in df.iterrows():
    d = r["Date"]
    w, l = r["Winner"], r["Loser"]

    player_dates[w].append(d)
    player_results[w].append(1)
    player_opponents[w].append(l)

    player_dates[l].append(d)
    player_results[l].append(0)
    player_opponents[l].append(w)

# ======================
# FEATURE HELPERS (IDENTICAL LOGIC)
# ======================
def last_n_win_rate(player, date, n=5):
    dates = player_dates[player]
    if not dates:
        return 0.5

    idx = bisect_left(dates, date)
    start = max(0, idx - n)
    recent = player_results[player][start:idx]
    return sum(recent) / len(recent) if recent else 0.5


def fatigue_matches(player, date, days=14):
    dates = player_dates[player]
    left = bisect_left(dates, date - pd.Timedelta(days=days))
    right = bisect_left(dates, date)
    return right - left


def h2h_win_pct(A, B, date):
    dates = player_dates[A]
    opps = player_opponents[A]
    results = player_results[A]

    idx = bisect_left(dates, date)
    h2h = [results[i] for i in range(idx) if opps[i] == B]
    return sum(h2h) / len(h2h) if h2h else 0.5


def get_surface_elo(player, surface):
    col = f"elo_{surface}"
    return elo_df.loc[player, col] if player in elo_df.index else 1500

# ======================
# BANKROLL / STRATEGY
# ======================
BANKROLL = 1000
FLAT_STAKE_PCT = 0.01
KELLY_FRACTION = 0.5
EV_THRESHOLD = 0.05

# ======================
# MANUAL POLYMARKET INPUT (MULTIPLE MATCHES)
# ======================
markets = [
    {
        "player_A": "Ben Shelton",
        "player_B": "Sebastian Baez",
        "price_yes": 0.77,
        "price_no": 0.25,
        "surface": "hard",
        "best_of": 3,
        "tournament_level": "250",
        "round_number": "Quarterfinals",          # Quarterfinals
        "match_date": "2026-01-14"
    },
    {
        "player_A": "Luciano Darderi",
        "player_B": "Marcos Giron",
        "price_yes": 0.32,          # Darderi 32c
        "price_no": 0.70,           # Giron 70c
        "surface": "hard",
        "best_of": 5,
        "tournament_level": "250",
        "round_number": "Quarterfinals",
        "match_date": "2025-01-14"
    },
    {
        "player_A": "Fabian Marozsan",
        "player_B": "Eliot Spizzirri",
        "price_yes": 0.55,          # Marozsan 55c
        "price_no": 0.46,           # Spizzirri 46c
        "surface": "hard",
        "best_of": 5,
        "tournament_level": "250",
        "round_number": "Quarterfinals",
        "match_date": "2025-01-14"
    }
    
    
]


# ======================
# EVALUATE MATCHES
# ======================
rows = []

for m in markets:
    A, B = m["player_A"], m["player_B"]
    date = pd.to_datetime(m["match_date"])
    surface = m["surface"]

    elo_A = get_surface_elo(A, surface)
    elo_B = get_surface_elo(B, surface)

    features = {
        "surface": surface,
        "tournament_level": m["tournament_level"],
        "round_number": m["round_number"],
        "elo_diff_surface": elo_A - elo_B,
        "elo_A_surface": elo_A,
        "elo_B_surface": elo_B,
        "last5_win_rate_diff": last_n_win_rate(A, date) - last_n_win_rate(B, date),
        "h2h_win_pct_diff": h2h_win_pct(A, B, date) - 0.5,
        "fatigue_diff": fatigue_matches(A, date) - fatigue_matches(B, date),
        "log_odds_diff": np.log(m["price_yes"]) - np.log(m["price_no"]),
        "odds_A": 1 / m["price_yes"],
        "odds_B": 1 / m["price_no"],
        "best_of": m["best_of"]
    }

    X = pd.DataFrame([features])
    p_raw = model.predict_proba(X)[:, 1]
    p = calibrator.predict_proba(p_raw.reshape(-1, 1))[0, 1]

    ev_yes = p - m["price_yes"]
    ev_no = (1 - p) - m["price_no"]

    flat_stake = BANKROLL * FLAT_STAKE_PCT
    kelly_frac = max((p - m["price_yes"]) / (1 - m["price_yes"]), 0)
    kelly_stake = BANKROLL * kelly_frac * KELLY_FRACTION

    decision = "NO BET"
    stake = 0

    if ev_yes >= EV_THRESHOLD:
        decision = "BET YES"
        stake = min(max(flat_stake, kelly_stake), BANKROLL * 0.03)

    rows.append({
        "Match": f"{A} vs {B}",
        "P(A wins)": round(p, 3),
        "YES price": m["price_yes"],
        "NO price": m["price_no"],
        "EV YES": round(ev_yes, 3),
        "EV NO": round(ev_no, 3),
        "Decision": decision,
        "Stake ($)": round(stake, 2)
    })

# ======================
# OUTPUT
# ======================
results = pd.DataFrame(rows)
print("\n=== POLYMARKET EV DECISIONS ===\n")
print(results.to_string(index=False))
