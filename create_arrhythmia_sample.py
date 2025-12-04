import pandas as pd

df = pd.read_csv("incart_arrhythmia.csv")

# filter abnormal (arrhythmia) beats
abnormal = df[df["type"] != "N"].head(200)

abnormal.to_csv("arrhythmia_segment.csv", index=False)
print("arrhythmia_segment.csv created!")
