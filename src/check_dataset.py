import pandas as pd

df = pd.read_csv("data/processed/model_dataset.csv")

print("Class balance:")
print(df["y"].value_counts(normalize=True))
print("Total rows:", len(df))
