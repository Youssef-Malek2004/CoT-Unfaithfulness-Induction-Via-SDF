import json
import random
from pathlib import Path

DATA_DIR = Path("../data/arc_easy")
SPLIT = "train"
N_SAMPLES = 5

with open(DATA_DIR / f"{SPLIT}.json", "r") as f:
    data = json.load(f)

# Sample
samples = random.sample(data, N_SAMPLES)

for i, ex in enumerate(samples, 1):
    print(f"\n=== Sample {i} ===")
    print("Question:", ex["question"])
    for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]):
        print(f"{l}) {t}")
    print("AnswerKey:", ex["answerKey"])
