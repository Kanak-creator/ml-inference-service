import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import random
import os
import sys

rows = []

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Positive class (label = 1)
for _ in range(500):
    f1 = random.uniform(60, 100)
    f2 = random.uniform(1, 50)
    rows.append([f1, f2, 1])

# Negative class (label = 0)
for _ in range(500):
    f1 = random.uniform(0, 40)
    f2 = random.uniform(-50, -1)
    rows.append([f1, f2, 0])

df = pd.DataFrame(rows, columns=["feature1", "feature2", "label"])
print("adding test")
print(df["label"].value_counts())
os.makedirs('data', exist_ok=True)
df.to_csv("data/data.csv", index=False)

# In real pipelines, data generation and training are separate steps
df = pd.read_csv("data/data.csv")


X = df[["feature1", "feature2"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"accuracy is {acc}")

os.makedirs(f"model/{VERSION}", exist_ok=True)
joblib.dump(model, f"model/{VERSION}/model.joblib")
print("model created")

metadata = {
    "version": VERSION,
    "accuracy": acc,
    "random_seed": RANDOM_SEED,
    "rows": len(df),
    "features": ["feature1", "feature2"],
    "model_type": "LogisticRegression"
}

with open(f"model/{VERSION}/metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("metrics created")

#temp code for verification
# loaded = joblib.load("model/model.joblib")
# print("Model type:", type(loaded))