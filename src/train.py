import torch
from torch import nn
from tqdm.auto import tqdm

from model import model
from data import train_data, train_label
from data import test_data, test_label
from config import EPOCHS, LR

# Create the Training and Testing function
def train_test_func(model:torch.nn.Module,
                    epochs: int,
                    loss_fn: torch.nn.L1Loss,
                    optimizer: torch.optim.Optimizer):
    for epoch in tqdm(range(epochs)):
        model.train()

        train_pred = model(train_data)
        train_loss = loss_fn(train_pred, train_label)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval() 
        with torch.inference_mode():
            test_pred = model(test_data)
            test_loss = loss_fn(test_pred, test_label)


        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train loss: {train_loss} | Test loss: {test_loss.item()}", end=" \n")


# Create loss function and optimizer
loss = nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

if __name__ == '__main__':
    train_test_func(model=model, epochs=EPOCHS, loss_fn=loss, optimizer=optimizer)
