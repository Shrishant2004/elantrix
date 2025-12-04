import pandas as pd

df = pd.read_csv("incart_arrhythmia.csv")

# filter normal beats
normal = df[df["type"] == "N"].head(200)   # take first 200 normal beats

normal.to_csv("normal_segment.csv", index=False)
print("normal_segment.csv created!")
