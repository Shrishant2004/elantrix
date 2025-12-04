import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ----- 1. Load your dataset -----
df = pd.read_csv("incart_arrhythmia.csv")  # <--- change if your CSV has another name

# ----- 2. Create binary label -----
df["label"] = (df["type"] != "N").astype(int)

# ----- 3. Clean data -----
df = df.dropna()

drop_cols = ["record", "type"]
feature_cols = [c for c in df.columns if c not in drop_cols + ["label"]]

X = df[feature_cols]
y = df["label"]

# ----- 4. Train-test split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- 5. Train the model -----
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----- 6. Evaluate -----
print("\nMODEL PERFORMANCE:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ----- 7. Save the model -----
joblib.dump(
    {"model": model, "features": feature_cols},
    "elantrix_arrhythmia_model.pkl"
)

print("\nModel saved as 'elantrix_arrhythmia_model.pkl'")
