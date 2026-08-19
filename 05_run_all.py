"""
Run this single file to execute the full ablation pipeline:
  1. Preprocess both datasets
  2. Train all 5 model variants
  3. Evaluate and generate ablation tables
"""

import subprocess, sys

steps = [
    '01_preprocessing.py',
    '03_train.py',
    '04_evaluate.py',
]

for step in steps:
    print(f"\n{'='*60}")
    print(f"▶ Running: {step}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, step], check=True)

print("\n🎉 Full ablation pipeline complete! Check results/ folder.")
