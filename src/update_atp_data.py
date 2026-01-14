import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2022, 2027)

def update_data():
    for year in YEARS:
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
        out_file = OUT_DIR / f"ATP_{year}.csv"

        try:
            # IMPORTANT: read the ATP sheet
            df = pd.read_excel(url, sheet_name=0)
            df.to_csv(out_file, index=False)
            print(f"✔ Updated ATP {year} ({len(df)} matches)")
        except Exception as e:
            print(f"✘ ATP {year} failed → {e}")

if __name__ == "__main__":
    update_data()








