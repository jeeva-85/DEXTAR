import sys
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ml.training.train_failure import train_failure_model

if __name__ == "__main__":
    print("Training M3: Failure Risk Model (Real Data)...")
    res = train_failure_model()
    print(f"M3 Training Complete. Status: {res.get('status')}, Metrics: {res.get('metrics')}")
