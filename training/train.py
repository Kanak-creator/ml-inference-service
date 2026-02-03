import pandas as pd
import numpy as np
import random
import json
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------
# Reproducibility
# -------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------
# Model version (CLI arg)
# -------------------
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v3"

# -------------------
# Load real dataset
# -------------------
df = pd.read_csv("data/titanic.csv")

# -------------------
# Select features & target
# -------------------
features = ["Pclass", "Sex", "Age", "Fare"]
target = "Survived"

X = df[features]
y = df[target]

# -------------------
# Preprocessing
# -------------------

# Encode Sex: male=0, female=1
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Fill missing Age with mean
X["Age"] = X["Age"].fillna(X["Age"].mean())

# Fill missing Fare (rare but safe)
X["Fare"] = X["Fare"].fillna(X["Fare"].mean())

# -------------------
# Train/test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# -------------------
# Train model
# -------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------
# Evaluate
# -------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc}")

# -------------------
# Save model artifact
# -------------------
os.makedirs(f"model/{VERSION}", exist_ok=True)
joblib.dump(model, f"model/{VERSION}/model.joblib")

# -------------------
# Save metadata
# -------------------
metadata = {
    "version": VERSION,
    "accuracy": acc,
    "random_seed": RANDOM_SEED,
    "rows": len(df),
    "features": features,
    "model_type": "LogisticRegression",
    "dataset": "Titanic (Kaggle)"
}

with open(f"model/{VERSION}/metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model {VERSION} trained and saved.")
