import json
from pathlib import Path

with open("configs/config.json", "r") as file:
    data = json.load(file)

TEST = data["test"]
LR = data["lr"]
RANDOM_SEED = data["random_seed"]
EPOCHS = data["num_epochs"]

PROJECT_ROOT = Path(__file__).parent.parent