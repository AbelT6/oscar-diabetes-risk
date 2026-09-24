from ucimlrepo import fetch_ucirepo
import pandas as pd

#fetch dataset from UCI ML Repository
diabetes = fetch_ucirepo(id=891)

#features (x) and target labels (y) come back as separate pd Df
x= diabetes.data.features
y= diabetes.data.targets

#combine them into one Df
df = pd.concat([x,y], axis=1)

#save locally so future scripts don't need re-fetch

df.to_csv("data/diabetes_raw.csv", index= False)

print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to data/diabetes_raw.csv")
print(df.head())




