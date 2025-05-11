# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class LSTMTrainer:
    def __init__(self, model, device="cpu", lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': []}
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze(-1)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, epochs=20):
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")
        return self.history