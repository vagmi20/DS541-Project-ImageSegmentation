import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from slic import SLICProcessor
from superpixelnetwork import SuperpixelNeuralNetwork
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration
IMAGE_PATH = "/Users/a981199/Downloads/test_images/00000003_007.png"
NUM_SUPERPIXELS = 100
COMPACTNESS = 20
HEIGHT, WIDTH, CHANNELS = 224, 224, 3
NUM_CLASSES = 10
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 0.001

print("ok")


# Step 1: Preprocess Images with slic.py
def preprocess_images(image_path):
    # Open image and ensure it's in RGB mode
    img = Image.open(image_path).convert("RGB")
    img = img.resize((WIDTH, HEIGHT))
    img.save("resized_image.png")

    # Run SLICProcessor
    slic_processor = SLICProcessor("resized_image.png", K=NUM_SUPERPIXELS, M=COMPACTNESS)
    labels = slic_processor.iterate_10times()

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
                A[labels[i, j] - 1, labels[i + 1, j] - 1] = 1
                A[labels[i + 1, j] - 1, labels[i, j] - 1] = 1
            if labels[i, j] != labels[i, j + 1]:
                A[labels[i, j] - 1, labels[i, j + 1] - 1] = 1
                A[labels[i, j + 1] - 1, labels[i, j] - 1] = 1

    # Aggregation Matrix
    Q = np.zeros((num_superpixels, h * w))
    for superpixel_id, cluster in enumerate(clusters):
        for r, c in cluster.pixels:
            Q[superpixel_id, r * w + c] = 1
        Q[superpixel_id] /= Q[superpixel_id].sum()  # Normalize

    return A, Q


# Step 2: Load Data
def load_data():
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    train_dataset = datasets.FakeData(transform=transform)  # Replace with your dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


# Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        print(f"Starting Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log batch progress
            print(f"Batch {batch_idx + 1}/{len(train_loader)} processed.")

        print(f"Epoch [{epoch + 1}/{EPOCHS}] completed. Loss: {total_loss / len(train_loader):.4f}")


# Main Execution
if __name__ == "__main__":
    # Step 1: Preprocess Images
    adjacency_matrix, aggregation_matrix = preprocess_images(IMAGE_PATH)

    # Step 2: Load Matrices into PyTorch Tensors
    A = torch.from_numpy(adjacency_matrix).float().to(device)
    Q = torch.from_numpy(aggregation_matrix).float().to(device)

    # Step 3: Initialize Model
    model = SuperpixelNeuralNetwork(HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, Q, A).to(device)
    print(model)

    # Step 4: Load Data and Train the Model
    train_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_model(model, train_loader, criterion, optimizer)
