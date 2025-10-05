import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import boto3
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Neural network with dynamic number of output classes
class ObesityNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ObesityNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # Output matches number of classes
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train function for a single client
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

# Function to save the model to S3
def save_model_to_s3(model, bucket_name, s3_key):
    # Create an S3 client
    s3 = boto3.client('s3')
    
    # Save the model locally first
    model_path = 'obesity_model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Upload the model to S3
    s3.upload_file(model_path, bucket_name, s3_key)
    print(f"Model uploaded to S3 bucket: {bucket_name}, with key: {s3_key}")

# Function to load dataset from S3
def load_dataset_from_s3(bucket_name, s3_key):
    # Create an S3 client
    s3 = boto3.client('s3')
    
    # Get the object from S3
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    data = obj['Body'].read().decode('utf-8')  # Read the file content and decode it
    
    # Load the CSV into pandas DataFrame
    dataset = pd.read_csv(StringIO(data))
    
    return dataset

# Function to evaluate the model on the test set
def evaluate_model(global_model, test_loader):
    global_model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient calculation (we're just evaluating)
        for data, target in test_loader:
            output = global_model(data)  # Forward pass
            _, predicted = torch.max(output.data, 1)  # Get the predicted class
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * sum([1 for p, t in zip(all_predictions, all_targets) if p == t]) / len(all_targets)
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,  # Convert to percentage
        'recall': recall * 100,
        'f1_score': f1 * 100
    }

if __name__ == "__main__":
    # Parse arguments for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients for federated learning")
    parser.add_argument("--optimizer", type=str, choices=["sgd","adam"], default="sgd", help="Optimizer to use on clients")
    parser.add_argument("--scale_features", action="store_true", help="Scale numeric features with StandardScaler")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights in CrossEntropyLoss")
    parser.add_argument("--s3_data_bucket", type=str, required=False, help="S3 bucket name where dataset is stored")
    parser.add_argument("--s3_data_key", type=str, required=False, help="S3 key (path) of the dataset CSV file")
    parser.add_argument("--s3_model_bucket", type=str, required=False, help="S3 bucket name where the model will be uploaded")
    parser.add_argument("--s3_model_key", type=str, required=False, help="S3 key (path) where the model will be uploaded")
    parser.add_argument("--local_data_path", type=str, required=False, help="Path to a local CSV dataset to use instead of S3")
    parser.add_argument("--local_model_path", type=str, required=False, help="Path to save the trained model locally instead of uploading to S3")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target (label) column in the dataset")
    args = parser.parse_args()

    # Load dataset from either local path or S3
    if args.local_data_path:
        print(f"Loading dataset from local path: {args.local_data_path}")
        data = pd.read_csv(args.local_data_path)
    else:
        if not args.s3_data_bucket or not args.s3_data_key:
            raise ValueError("Either --local_data_path or both --s3_data_bucket and --s3_data_key must be provided to load the dataset.")
        data = load_dataset_from_s3(args.s3_data_bucket, args.s3_data_key)

    # Separate features and labels
    if args.target_column not in data.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in the dataset.")

    # Label encode categorical columns
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    # Optionally scale numeric features (fit on full data to keep things simple)
    if args.scale_features:
        print("Scaling numeric features with StandardScaler")
        scaler = StandardScaler()
        # Only scale numeric columns (non-integer encoded object columns are already encoded above)
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        # Exclude the target column from scaling to avoid transforming labels
        if args.target_column in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != args.target_column]
        if numeric_cols:
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Split features and labels
    X = data.drop(args.target_column, axis=1).values
    y = data[args.target_column].values

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors with appropriate types
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Ensure y_train is of type long
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # Ensure y_test is of type long

    # Create TensorDataset for test data
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Split training data between simulated clients
    num_clients = args.num_clients
    X_train_chunks = torch.chunk(X_train_tensor, num_clients)
    y_train_chunks = torch.chunk(y_train_tensor, num_clients)

    # Create TensorDatasets and DataLoaders for each client
    client_data = [TensorDataset(X_train_chunks[i], y_train_chunks[i]) for i in range(num_clients)]
    client_loaders = [DataLoader(subset, batch_size=32, shuffle=True) for subset in client_data]

    # Dynamically determine the number of classes
    num_classes = len(data[args.target_column].unique())

    # Instantiate the global model and define loss function
    input_size = X_train.shape[1]
    global_model = ObesityNet(input_size, num_classes)
    # Optionally compute class weights
    if args.use_class_weights:
        # compute weights inversely proportional to class frequency
        import numpy as _np
        classes, counts = _np.unique(y_train, return_counts=True)
        weights = _np.sum(counts) / (len(classes) * counts)
        weights = torch.tensor(weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: {weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Federated learning parameters
    federated_accuracies = []
    for epoch in range(args.epochs):
        client_models = []
        client_optimizers = []
        client_sizes = [len(chunk) for chunk in X_train_chunks]
        for _ in range(num_clients):
            # Clone the global model for each client
            client_model = ObesityNet(input_size, num_classes)
            client_model.load_state_dict(global_model.state_dict())
            client_models.append(client_model)
            if args.optimizer == 'adam':
                client_optimizers.append(optim.Adam(client_model.parameters(), lr=args.learning_rate))
            else:
                client_optimizers.append(optim.SGD(client_model.parameters(), lr=args.learning_rate))
        
        # Train on each client's data independently
        client_losses = []
        for i, client_loader in enumerate(client_loaders):
            client_loss = train_on_client(client_loader, client_models[i], criterion, client_optimizers[i])
            client_losses.append(client_loss)
        
        # Aggregate client models to update the global model
        # Support federated averaging weighted by client sample size
        global_state_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        for key in global_state_dict.keys():
            stacked = torch.stack([client_models[i].state_dict()[key].float() for i in range(num_clients)])
            # compute weighted average along dim=0
            weights = torch.tensor([client_sizes[i] for i in range(num_clients)], dtype=torch.float32)
            weights = weights / weights.sum()
            # reshape weights for broadcasting
            while weights.dim() < stacked.dim():
                weights = weights.unsqueeze(-1)
            global_state_dict[key] = (stacked * weights).sum(0)
        global_model.load_state_dict(global_state_dict)
        metrics = evaluate_model(global_model, test_loader)
        federated_accuracies.append(metrics["accuracy"])

        print(f'Epoch {epoch + 1}, Client Losses: {client_losses}, Global Model Loss: {sum(client_losses) / num_clients}, Accuracy: {metrics["accuracy"]:.2f}%')

    # Save the model either locally or to S3 after training
    if args.local_model_path:
        torch.save(global_model.state_dict(), args.local_model_path)
        print(f"Model saved locally at: {args.local_model_path}")
    else:
        if not args.s3_model_bucket or not args.s3_model_key:
            raise ValueError("Either --local_model_path or both --s3_model_bucket and --s3_model_key must be provided to save the model.")
        save_model_to_s3(global_model, args.s3_model_bucket, args.s3_model_key)

    #Non - Federated learning Method (for accuracy Comparison)
    non_federated_model = ObesityNet(input_size, num_classes)
    optimizer = optim.SGD(non_federated_model.parameters(), lr=args.learning_rate)

    non_federated_accuracies = []  
    full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(args.epochs):
        non_federated_model.train()
        epoch_loss = 0
        for data, target in full_train_loader:
            optimizer.zero_grad()
            output = non_federated_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        metrics_nf = evaluate_model(non_federated_model, test_loader)
        non_federated_accuracies.append(metrics_nf["accuracy"])

        if (epoch + 1) % 100 == 0:
            print(f'Non-Federated Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss/len(full_train_loader):.4f}, Accuracy: {metrics_nf["accuracy"]:.2f}%')

    federated_accuracy = federated_accuracies[-1]
    non_federated_accuracy = non_federated_accuracies[-1]
    # Evaluate the federated global model on the test set
    metrics = evaluate_model(global_model, test_loader)
    
         
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f'Accuracy:  {metrics["accuracy"]:.2f}%')
    print(f'Precision: {metrics["precision"]:.2f}%')
    print(f'Recall:    {metrics["recall"]:.2f}%')
    print(f'F1-Score:  {metrics["f1_score"]:.2f}%')
    print("="*60)

    print(f"\nFinal Federated Model Accuracy: {federated_accuracy:.2f}%")
    print(f"Final Non-Federated Model Accuracy: {non_federated_accuracy:.2f}%")

     # --- Plotting Accuracy Comparison ---
    plt.figure(figsize=(12, 7))
    epochs_range = range(1, args.epochs + 1)

    plt.plot(epochs_range, federated_accuracies, 'r.-', label='Federated Accuracy')
    plt.plot(epochs_range, non_federated_accuracies, 'b.-', label='Non-Federated Accuracy')

    title_str = (f'Accuracy Comparison (Federated vs Non-Federated)\n'
                 f'(Final Federated: {federated_accuracy:.2f}%, Non-Federated: {non_federated_accuracy:.2f}%)')
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.show()


    #Plotting Federated vs Non-Federated Accuracy from 3 Random Clients
    client_indices = random.sample(range(num_clients), min(3, num_clients))

    for client_id in client_indices:
        print(f"\nTraining Non-Federated Model for Client {client_id+1}...")

        # Create client-specific dataset
        client_dataset = TensorDataset(X_train_chunks[client_id], y_train_chunks[client_id])
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

        # Re-initialize a fresh model for this client
        client_model = ObesityNet(input_size, num_classes)
        client_optimizer = optim.SGD(client_model.parameters(), lr=args.learning_rate)
        
        # Track accuracies
        client_nonfl_accuracies = []
        
        for epoch in range(args.epochs):
            client_model.train()
            for data, target in client_loader:
                client_optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                client_optimizer.step()
            
            # Evaluate client model on the global test set
            metrics_client = evaluate_model(client_model, test_loader)
            client_nonfl_accuracies.append(metrics_client["accuracy"])
    
        # --- Plotting for this client ---
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, args.epochs + 1)

        plt.plot(epochs_range, federated_accuracies, 'r-', label='Federated (Global)')
        plt.plot(epochs_range, client_nonfl_accuracies, 'b-', label=f'Client {client_id+1} Non-FL')

        title_str = (f'Client {client_id+1}: Federated vs Non-Federated Accuracy\n'
                    f'(Final FL: {federated_accuracies[-1]:.2f}%, '
                    f'Client Non-FL: {client_nonfl_accuracies[-1]:.2f}%)')
        plt.title(title_str)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.show()


