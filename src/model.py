import torch
import torch.nn as nn

# Create the model class
class Delivery_timesModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.Linear = nn.Sequential(
            nn.Linear(16, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, 1)
        )

    def forward(self, x):
        return self.Linear(x)

# Create the model
torch.manual_seed(42)

model = Delivery_timesModel()