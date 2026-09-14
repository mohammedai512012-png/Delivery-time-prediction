# Import libraries
import torch
import torch.nn as nn

from pathlib import Path
from model import Delivery_timesModel

from train import train_test_func
from helper_functions import cmd_cleaner, save_model
from config import LR, EPOCHS, PROJECT_ROOT

cmd_cleaner(work_by="cmd", clean=True)

print(f"PyTorch version: {torch.__version__}")

# Create the model
model = Delivery_timesModel()

# Create loss function and optimizer
loss = nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)


train_test_func(model=model,
                epochs=EPOCHS,
                loss_fn=loss,
                optimizer=optimizer)

# Save the model
MODEL_PATH = Path(f"{PROJECT_ROOT}/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DeliveryModelV0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

save_model(model, MODEL_SAVE_PATH)
