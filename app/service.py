print(">>> SERVICE.PY LOADED <<<")

import joblib
import logging
import time
import os
import pandas as pd

logger = logging.getLogger()

class ModelService:
    def __init__(self):
        print(">>> MODEL INIT <<<")
        MODEL_VERSION = os.getenv("MODEL_VERSION", "v3")
        try:
            start = time.time()
            self.model = joblib.load(f"model/{MODEL_VERSION}/model.joblib")
            elapsed = (time.time() - start) * 1000
            logger.info(f"Model loaded successfully in {elapsed:.2f} ms")
        except Exception as e:
            logger.error("Failed to load model", exc_info=e)
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        df = pd.DataFrame([features], columns=["Pclass", "Sex", "Age", "Fare"])
        result = self.model.predict_proba(df)[0][1]
        latency_ms = (time.time() - start) * 1000

        logger.info(
            "Prediction completed",
            extra={
                "features": features,
                "latency_ms": round(latency_ms, 2),
            },
        )

        return float(result)
