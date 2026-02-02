Week 2 — Model Training & Artifacts
Purpose

This module demonstrates a reproducible ML training workflow aligned with a production API contract.

The goal of Week 2 is not model optimization, but to:

Validate the training → artifact → inference lifecycle

Produce a versioned, deployable model artifact

Keep training logic decoupled from serving logic

Scope (What This Week Covers)

✅ Synthetic data generation aligned with API schema
✅ Reproducible model training
✅ Train/test split with fixed random seed
✅ Model artifact serialization
✅ Metrics persistence

❌ No notebooks
❌ No feature engineering complexity
❌ No AWS/SageMaker yet (local-first)

Training Data Schema

The training dataset (data/data.csv) uses a fixed schema that matches the Week-1 API contract:

feature1,feature2,label

Column Definitions
Column	Type	Description
feature1	float	Numeric business signal (0–100)
feature2	float	Numeric business signal (-50–50)
label	int	Binary outcome (1 = positive, 0 = negative)
Design Rationale

Feature ranges exactly match API validation

Unknown fields are forbidden at the API layer

Prevents training–serving skew by design

The intent is to validate the pipeline, not overfit to a dataset.

Synthetic Data Generation

For this prototype, training data is synthetically generated using a deterministic rule:

label = 1 if feature1 > 50 and feature2 > 0 else 0

Why Synthetic Data?

No dependency on sensitive or proprietary datasets

Fully reproducible

Allows focus on system correctness, not data sourcing

In real systems, this step would be replaced by:

Batch ingestion

Feature pipelines

Offline feature stores

Training Script
Entry Point
python training/train.py

What the Script Does

Generates (or reads) training data

Splits data into train/test sets

Trains a Logistic Regression model

Evaluates accuracy

Writes artifacts to disk

Reproducibility

Fixed random_state in train/test split

Deterministic synthetic data logic

Identical runs produce identical artifacts

Model Artifacts

After a successful run, the following files are created:

ml-service/
├── model/
│   ├── model.joblib
│   └── metrics.json

model.joblib

Serialized scikit-learn model

Intended to be loaded once at service startup

Treated as an immutable artifact

metrics.json

Example:

{
  "accuracy": 0.82
}


Metrics are persisted to:

Enable offline evaluation

Support future model comparison

Feed deployment decisions in later stages

Separation of Concerns

This project intentionally separates:

Concern	Responsibility
Data generation	Training pipeline
Model training	Offline process
Model serving	API service (Week 3)
Validation	API contract (Week 1)

Training code should never live inside API handlers.

Known Limitations (Intentional)

Accuracy is the only metric

No hyperparameter tuning

No cross-validation

No feature scaling

These will be addressed only when the system requires it.

How This Evolves Next

In upcoming weeks, this training workflow will be:

Containerized

Executed in SageMaker

Triggered by data drift

Integrated with MLOps pipelines

The core contract and artifacts will remain unchanged.

Interview Talking Points

If asked about this stage, the correct framing is:

“I focused on making the training pipeline reproducible and aligned with the API contract. The goal was to produce deployable artifacts, not maximize accuracy.”

That answer is exactly right.

Status

✔ Week 2 complete
