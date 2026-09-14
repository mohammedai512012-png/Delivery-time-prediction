import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST, RANDOM_SEED, PROJECT_ROOT

row_data = pd.read_csv(f"{PROJECT_ROOT}/data/data.csv")

data = row_data.iloc[:, 1:-1].to_numpy()
labels = row_data.iloc[:, -1].to_numpy()

# Split the data into training and test data
train_data, test_data, train_label, test_label = train_test_split(
    data,
    labels,
    test_size=TEST,
    random_state=RANDOM_SEED
)

# Create standard scaler
TrainScaler = StandardScaler()
TestScaler = StandardScaler()

# Scale data & Turn it into tensors
train_data = torch.tensor(TrainScaler.fit_transform(train_data), dtype=torch.float32)
test_data  = torch.tensor(TrainScaler.transform(test_data),      dtype=torch.float32)

train_label = torch.tensor(TestScaler.fit_transform(train_label.reshape(-1,1)), dtype=torch.float32)
test_label  = torch.tensor(TestScaler.transform(test_label.reshape(-1,1)),      dtype=torch.float32)