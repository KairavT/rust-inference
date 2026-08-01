import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(in_features=4, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=3)
)

optimizer = optim.Adam(model.parameters(), lr = 5e-3)
loss_fn = nn.CrossEntropyLoss()
for i in range(1001):
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 100 == 0:
        print(f'Iteration {i}, loss = {loss:.5f}')