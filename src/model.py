import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left

# ======================
# FILE PATHS
# ======================
MATCHES_FILE = Path("data/processed/atp_matches_all.csv")
ELO_FILE = Path("data/external/ta_atp_elo_surfaces.csv")
OUT_FILE = Path("data/processed/model_dataset.csv")

# ======================
# LOAD MATCH DATA
# ======================
df = pd.read_csv(MATCHES_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Winner"] = df["Winner"].astype(str).str.strip()
df["Loser"] = df["Loser"].astype(str).str.strip()

df = df.dropna(subset=["Date", "Winner", "Loser", "AvgW", "AvgL"])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# LOAD ELO DATA
# ======================
elo_df = pd.read_csv(ELO_FILE)
elo_df["player"] = elo_df["player"].astype(str).str.strip()
elo_df = elo_df.set_index("player")

# ======================
# PRECOMPUTE PLAYER HISTORY (FAST)
# ======================
player_dates = defaultdict(list)
player_results = defaultdict(list)     # 1 = win, 0 = loss
player_opponents = defaultdict(list)

for _, r in df.iterrows():
    date = r["Date"]
    w, l = r["Winner"], r["Loser"]

    player_dates[w].append(date)
    player_results[w].append(1)
    player_opponents[w].append(l)

    player_dates[l].append(date)
    player_results[l].append(0)
    player_opponents[l].append(w)

# ======================
# HELPER FUNCTIONS (O(log N))
# ======================
def last_n_win_rate(player, date, n=5):
    dates = player_dates[player]
    if not dates:
        return 0.5

    idx = bisect_left(dates, date)
    if idx == 0:
        return 0.5

    start = max(0, idx - n)
    recent = player_results[player][start:idx]

    return sum(recent) / len(recent) if recent else 0.5


def fatigue_matches(player, date, days=14):
    dates = player_dates[player]
    if not dates:
        return 0

    left = bisect_left(dates, date - pd.Timedelta(days=days))
    right = bisect_left(dates, date)
    return right - left


def h2h_win_pct(A, B, date):
    dates = player_dates[A]
    opps = player_opponents[A]
    results = player_results[A]

    idx = bisect_left(dates, date)

    h2h = [
        results[i]
        for i in range(idx)
        if opps[i] == B
    ]

    return sum(h2h) / len(h2h) if h2h else 0.5

# ======================
# FEATURE BUILDER
# ======================
def build_row(A, B, odds_A, odds_B, y, row):
    date = row["Date"]
    surface = str(row["Surface"]).lower()

    elo_A = elo_df.loc[A, f"elo_{surface}"] if A in elo_df.index else 1500
    elo_B = elo_df.loc[B, f"elo_{surface}"] if B in elo_df.index else 1500

    return {
        "elo_diff_surface": elo_A - elo_B,
        "elo_A_surface": elo_A,
        "elo_B_surface": elo_B,
        "last5_win_rate_diff": last_n_win_rate(A, date) - last_n_win_rate(B, date),
        "h2h_win_pct_diff": h2h_win_pct(A, B, date) - 0.5,
        "fatigue_diff": fatigue_matches(A, date) - fatigue_matches(B, date),
        "log_odds_diff": np.log(odds_A) - np.log(odds_B),
        "odds_A": odds_A,
        "odds_B": odds_B,
        "surface": surface,
        "best_of": row.get("Best of", 3),
        "tournament_level": row.get("Series", "250"),
        "round_number": row.get("Round", 1),
        "y": y
    }

# ======================
# BUILD DATASET
# ======================
rows = []

for _, row in df.iterrows():
    try:
        w, l = row["Winner"], row["Loser"]
        ow, ol = float(row["AvgW"]), float(row["AvgL"])

        if ow <= 1 or ol <= 1:
            continue

        # Winner perspective
        rows.append(build_row(w, l, ow, ol, 1, row))

        # Loser perspective
        rows.append(build_row(l, w, ol, ow, 0, row))

    except Exception:
        continue

# ======================
# SAVE DATASET
# ======================
model_df = pd.DataFrame(rows).dropna()

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
model_df.to_csv(OUT_FILE, index=False)

print("✔ Model dataset saved →", OUT_FILE)
print("Rows:", len(model_df))
print("Class balance:")
print(model_df["y"].value_counts(normalize=True))




