import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('ObesityDataSet.csv')

# Preprocess the data
data['Age'] = data['Age'].round().astype(int)
data['Height'] = data['Height'].round(2)
data['Weight'] = data['Weight'].round().astype(int)
data['FCVC'] = data['FCVC'].round().astype(int)
data['NCP'] = data['NCP'].round().astype(int)
data['CH2O'] = data['CH2O'].round().astype(int)
data['FAF'] = data['FAF'].round().astype(int)
data['TUE'] = data['TUE'].round().astype(int)

# Label encode categorical columns
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Split features and labels
X = data.drop('NObeyesdad', axis=1).values
y = data['NObeyesdad'].values

# Adjust labels to avoid having 0 as a label
# X += 1
# y += 1

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors with appropriate types
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDataset for test data
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Split training data between simulated clients (subsets)
num_clients = 5  # Adjust this to simulate more clients
X_train_chunks = torch.chunk(X_train_tensor, num_clients)
y_train_chunks = torch.chunk(y_train_tensor, num_clients)

# Create TensorDatasets and DataLoaders for each client
client_data = [TensorDataset(X_train_chunks[i], y_train_chunks[i]) for i in range(num_clients)]
client_loaders = [DataLoader(subset, batch_size=32, shuffle=True) for subset in client_data]

# Define a simple neural network
class ObesityNet(nn.Module):
    def __init__(self):
        super(ObesityNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)  # 7 output classes (NObeyesdad categories)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the global model and define loss function and optimizer
global_model = ObesityNet()
criterion = nn.CrossEntropyLoss()

# Function to train on client data
def train_on_client(client_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target in client_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(client_loader)

# Federated learning parameters
epochs = 2000  # Adjust the number to prevent overfitting
learning_rate = 0.01
for epoch in range(epochs):
    client_models = []
    client_optimizers = []
    for _ in range(num_clients):
        # Clone the global model for each client
        client_model = ObesityNet()
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)
        client_optimizers.append(optim.SGD(client_model.parameters(), lr=learning_rate))
    
    # Train on each client's data independently
    client_losses = []
    for i, client_loader in enumerate(client_loaders):
        client_loss = train_on_client(client_loader, client_models[i], criterion, client_optimizers[i])
        client_losses.append(client_loss)
    
    # Aggregate client models to update the global model
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        # Average each parameter across client models
        global_state_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(num_clients)]).mean(0)
    global_model.load_state_dict(global_state_dict)

    print(f'Epoch {epoch + 1}, Client Losses: {client_losses}, Global Model Loss: {sum(client_losses) / num_clients}')

# Evaluate the federated global model on the test set
global_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dataset:
        output = global_model(data.unsqueeze(0))  # Ensure data is batched
        _, predicted = torch.max(output.data, 1)
        total += 1
        correct += (predicted == target).sum().item()

print(f'Federated Learning Model Test Accuracy: {100 * correct / total:.2f}%')
