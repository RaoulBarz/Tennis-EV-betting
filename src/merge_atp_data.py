import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def merge_atp():
    frames = []

    for file in sorted(RAW_DIR.glob("ATP_*.csv")):
        year = int(file.stem.split("_")[1])
        df = pd.read_csv(file)

        df["season"] = year
        frames.append(df)

        print(f"Loaded {file.name} ({len(df)})")

    merged = pd.concat(frames, ignore_index=True)

    # optional cleanup
    merged = merged.drop_duplicates()

    out_file = OUT_DIR / "atp_matches_all.csv"
    merged.to_csv(out_file, index=False)

    print(f"✔ Merged dataset saved → {out_file}")

    print(f"Total matches: {len(merged)}")

if __name__ == "__main__":
    merge_atp()
