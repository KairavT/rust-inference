import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import json

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
    
weights_and_biases = {
    "hidden_weights": model[0].weight.detach().numpy().tolist(),
    "hidden_bias":model[0].bias.detach().numpy().tolist(),
    "output_weights": model[2].weight.detach().numpy().tolist(),
    "output_bias":model[2].bias.detach().numpy().tolist()
}

with open('weights.json', 'w') as f:
    json.dump(weights_and_biases, f)

iris_row_indices = [1, 50, 100, 2, 51]
indices_values = X[iris_row_indices]
with torch.no_grad():
    values_outputs = model(indices_values)

outputs_preds = values_outputs.argmax(dim=1)
print(outputs_preds)

values_to_preds = {
    "inputs":indices_values.tolist(),
    "predictions": outputs_preds.tolist()
}

with open('test_samples.json', 'w') as f:
    json.dump(values_to_preds, f)
