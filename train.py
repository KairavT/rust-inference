import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


model = nn.Sequential(
    nn.Linear(in_features=4, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=3)
)

optimizer = optim.Adam(model.parameters(), lr = 5e-3)
loss_fn = nn.CrossEntropyLoss()
for i in range(1001):
    out = model(X_train)
    loss = loss_fn(out, y_train)
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

with torch.no_grad():
    values_outputs = model(X_test)

with torch.no_grad():
    train_preds = model(X_train).argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()

    test_preds = model(X_test).argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item()

print(f'train accuracy: {train_acc} \
      test accuracy: {test_acc}')
      

outputs_preds = values_outputs.argmax(dim=1)

values_to_preds = {
    "inputs":X_test.tolist(),
    "predictions": outputs_preds.tolist()
}

with open('test_samples.json', 'w') as f:
    json.dump(values_to_preds, f)
