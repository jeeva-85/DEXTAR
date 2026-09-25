import sys
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ml.training.train_priority import train_priority_model

if __name__ == "__main__":
    print("Training M1: Maintenance Priority Model (Real Data)...")
    res = train_priority_model()
    print(f"M1 Training Complete. Status: {res.get('status')}, Metrics: {res.get('metrics')}")
