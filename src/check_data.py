import pandas as pd

df = pd.read_csv(r"dataset\machine_data.csv")

print("Dataset Loaded Successfully\n")

print("Shape:", df.shape)

print("\nColumns:\n", df.columns)

print("\nFirst 5 rows:\n", df.head())

print("\nMissing values:\n", df.isnull().sum())