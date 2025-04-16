import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SentimentTransformer

# Dummy dataset class (replace with your actual dataset)
class SampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # list of tokenized sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx])

# Hyperparameters
vocab_size = 10000
embed_dim = 128
num_heads = 4
hidden_dim = 256
num_classes = 2
lr = 1e-4
epochs = 5

# Dummy data
X_train = [[1, 234, 23, 4, 0, 0, 0], [5, 345, 2, 65, 0, 0, 0]] * 100
y_train = [1, 0] * 100

dataset = SampleDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "sentiment_model.pth")
