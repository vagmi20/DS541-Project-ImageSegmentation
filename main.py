import torch
import torch.optim as optim
import torch.nn as nn
from slic import SLICProcessor
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SuperpixelNeuralNetwork  # Import your model class

# Configuration
IMAGE_PATH = "example_image.jpg"
NUM_SUPERPIXELS = 100
COMPACTNESS = 20
HEIGHT, WIDTH, CHANNELS = 224, 224, 3
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# Step 1: Preprocess Images with slic.py
def preprocess_images(image_path):
    slic_processor = SLICProcessor(image_path, K=NUM_SUPERPIXELS, M=COMPACTNESS)
    slic_processor.iterate_10times()
    
    labels = np.zeros((slic_processor.image_height, slic_processor.image_width), dtype=int)
    for cluster in slic_processor.clusters:
        for (h, w) in cluster.pixels:
            labels[h, w] = cluster.no
    
    # Compute adjacency matrix (A) and aggregation matrix (Q)
    A, Q = compute_matrices(labels, slic_processor.clusters)
    np.save("adjacency_matrix.npy", A)
    np.save("aggregation_matrix.npy", Q)
    print("Superpixel preprocessing complete!")
    return A, Q

def compute_matrices(labels, clusters):
    """Compute adjacency and aggregation matrices based on superpixel labels."""
    num_superpixels = len(clusters)
    h, w = labels.shape
    
    # Adjacency Matrix
    A = np.zeros((num_superpixels, num_superpixels))
    for i in range(h - 1):
        for j in range(w - 1):
            if labels[i, j] != labels[i + 1, j]:
                A[labels[i, j], labels[i + 1, j]] = 1
                A[labels[i + 1, j], labels[i, j]] = 1
            if labels[i, j] != labels[i, j + 1]:
                A[labels[i, j], labels[i, j + 1]] = 1
                A[labels[i, j + 1], labels[i, j]] = 1
    
    # Aggregation Matrix
    Q = np.zeros((num_superpixels, h * w))
    for superpixel_id, cluster in enumerate(clusters):
        for h, w in cluster.pixels:
            Q[superpixel_id, h * w + w] = 1
        Q[superpixel_id] /= Q[superpixel_id].sum()
    
    return A, Q

# Step 2: Load Data
def load_data():
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.FakeData(transform=transform)  # Replace with your dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

# Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Main Execution
if __name__ == "__main__":
    # Preprocess Images
    adjacency_matrix, aggregation_matrix = preprocess_images(IMAGE_PATH)
    
    # Load Superpixel Matrices
    A = torch.from_numpy(adjacency_matrix).float().to(device)
    Q = torch.from_numpy(aggregation_matrix).float().to(device)
    
    # Initialize Model
    model = SuperpixelNeuralNetwork(HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, Q, A).to(device)
    print(model)
    
    # Load Data
    train_loader = load_data()
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the Model
    train_model(model, train_loader, criterion, optimizer)
