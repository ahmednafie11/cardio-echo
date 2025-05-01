import echonet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from echonet.models.r2plus1d import get_resnet_model

def train():
    dataset = echonet.datasets.Echo(root="~/cardio-echo/data", 
split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
num_workers=4)
    model = get_resnet_model()
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    for epoch in range(10):
        model.train()
        for batch in dataloader:
            frames = batch["frame"].cuda() if torch.cuda.is_available() 
else batch["frame"]
            ef = batch["ef"].cuda() if torch.cuda.is_available() else 
batch["ef"]
            optimizer.zero_grad()
            output = model(frames).squeeze()
            loss = criterion(output, ef)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
