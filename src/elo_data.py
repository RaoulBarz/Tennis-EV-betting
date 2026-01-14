import pandas as pd
from pathlib import Path
import requests
from io import StringIO

URL = "https://tennisabstract.com/reports/atp_elo_ratings.html"
OUT = Path("data/external/ta_atp_elo_surfaces.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Fetching data from: {URL}")
    
    # Fetch with proper headers
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"✓ Page fetched successfully")
        
    except Exception as e:
        print(f"✗ Network error: {e}")
        return
    
    # Parse all tables
    try:
        dfs = pd.read_html(StringIO(response.text))
        print(f"✓ Found {len(dfs)} table(s)")
    except Exception as e:
        print(f"✗ Parsing error: {e}")
        return
    
    # Find the right table
    df = None
    for i, table in enumerate(dfs):
        print(f"\nTable {i}: {table.shape}")
        print(f"Columns: {table.columns.tolist()[:10]}")  # Show first 10 columns
        
        # Look for table with Player column and Elo-related columns
        cols_str = str(table.columns).lower()
        if 'player' in cols_str or table.shape[1] >= 10:  # ATP table has many columns
            df = table
            print(f"→ Using this table")
            break
    
    if df is None:
        print("✗ Could not find the ATP ratings table")
        return
    
    print(f"\n📊 Working with table shape: {df.shape}")
    print(f"📋 Raw columns: {df.columns.tolist()}")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        print("Flattening MultiIndex columns...")
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns]
    
    # Convert to string and normalize
    df.columns = [str(c).strip() for c in df.columns]
    print(f"📋 Cleaned columns: {df.columns.tolist()}")
    
    # Find columns by checking actual column names
    # The table structure from tennisabstract typically has these columns:
    # Elo Rank, Player, Age, Elo, hElo Rank, hElo, cElo Rank, cElo, gElo Rank, gElo, ...
    
    player_col = None
    elo_col = None
    helo_col = None
    celo_col = None
    gelo_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        
        if player_col is None and 'player' in col_lower:
            player_col = col
            print(f"✓ Found Player: '{col}'")
        
        # Look for exact column names (Tennis Abstract uses these)
        if elo_col is None and col in ['Elo', 'elo']:
            elo_col = col
            print(f"✓ Found Elo: '{col}'")
        
        if helo_col is None and col in ['hElo', 'helo']:
            helo_col = col
            print(f"✓ Found hElo: '{col}'")
            
        if celo_col is None and col in ['cElo', 'celo']:
            celo_col = col
            print(f"✓ Found cElo: '{col}'")
            
        if gelo_col is None and col in ['gElo', 'gelo']:
            gelo_col = col
            print(f"✓ Found gElo: '{col}'")
    
    # Check what we found
    missing = []
    if player_col is None:
        missing.append("Player")
    if elo_col is None:
        missing.append("Elo")
    if helo_col is None:
        missing.append("hElo")
    if celo_col is None:
        missing.append("cElo")
    if gelo_col is None:
        missing.append("gElo")
    
    if missing:
        print(f"\n✗ Missing columns: {missing}")
        print(f"\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"  [{i}] '{col}'")
        
        # Try to manually identify columns by position if names don't match
        print("\n⚠️  Attempting to identify columns by position...")
        print("First few rows of data:")
        print(df.head(3))
        
        # Common pattern: [Elo Rank, Player, Age, Elo, hElo Rank, hElo, cElo Rank, cElo, gElo Rank, gElo]
        if df.shape[1] >= 10:
            player_col = df.columns[1]  # Usually column 1
            elo_col = df.columns[3]     # Usually column 3
            helo_col = df.columns[5]    # Usually column 5
            celo_col = df.columns[7]    # Usually column 7
            gelo_col = df.columns[9]    # Usually column 9
            print(f"Using positional mapping:")
            print(f"  Player: column {1} = '{player_col}'")
            print(f"  Elo: column {3} = '{elo_col}'")
            print(f"  hElo: column {5} = '{helo_col}'")
            print(f"  cElo: column {7} = '{celo_col}'")
            print(f"  gElo: column {9} = '{gelo_col}'")
    
    # Extract and save
    try:
        elo_df = df[[player_col, elo_col, helo_col, celo_col, gelo_col]].copy()
        elo_df.columns = ['player', 'elo_overall', 'elo_hard', 'elo_clay', 'elo_grass']
        
        # Clean data - remove any rows with missing player names
        elo_df = elo_df[elo_df['player'].notna()]
        
        # Convert Elo columns to numeric
        for col in ['elo_overall', 'elo_hard', 'elo_clay', 'elo_grass']:
            elo_df[col] = pd.to_numeric(elo_df[col], errors='coerce')
        
        elo_df.to_csv(OUT, index=False)
        print(f"\n✅ Success! Saved {len(elo_df)} players → {OUT}")
        print(f"\nTop 10 players:")
        print(elo_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Error extracting data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



