from datasets import load_dataset
import json
from pathlib import Path

dataset = load_dataset("hendrycks/competition_math")
print(dataset)

ex = dataset["train"][0]
print(ex)

out_dir = Path("../data/competition_math")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation", "test"]:
    with open(out_dir / f"{split}.json", "w") as f:
        json.dump(dataset[split].to_list(), f, indent=2)