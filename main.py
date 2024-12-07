import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from slic import SLICProcessor
from superpixelnetwork import SuperpixelNeuralNetwork
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration

#update this path 
DATASET_PATH = "/Users/a981199/Downloads/test_images"  
#you can change all the parameters to see the performance
NUM_SUPERPIXELS = 20
COMPACTNESS = 20
HEIGHT, WIDTH, CHANNELS = 224, 224, 3
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.01

# Image transformations
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open and transform the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


#  Load Dataset
def load_dataset(dataset_path):
    image_paths = []
    labels = []

    # Traverse dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

                # Extract label from folder name
                label = os.path.basename(root)
                labels.append(label)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return image_paths, labels


#  Preprocess Images with SLIC
def preprocess_images(image_paths):
    adjacency_matrix, aggregation_matrix = [], []

    for path in image_paths:
        # Preprocess each image
        slic_processor = SLICProcessor(path, K=NUM_SUPERPIXELS, M=COMPACTNESS)
        labels = slic_processor.iterate_Ntimes()

        # Compute adjacency matrix (A) and aggregation matrix (Q)
        A, Q = compute_matrices(labels, slic_processor.clusters)
        adjacency_matrix.append(A)
        aggregation_matrix.append(Q)

    return adjacency_matrix, aggregation_matrix


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


#  Train the Model
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
            print(f"Batch {batch_idx + 1}/{len(train_loader)} processed.")

        print(f"Epoch [{epoch + 1}/{EPOCHS}] completed. Loss: {total_loss / len(train_loader):.4f}")



if __name__ == "__main__":
    #
    image_paths, labels = load_dataset(DATASET_PATH)
    print(f"Loaded {len(image_paths)} images.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Preprocess with SLIC
    adjacency_matrix, aggregation_matrix = preprocess_images(X_train)
    A = torch.from_numpy(np.mean(adjacency_matrix, axis=0)).float().to(device)
    Q = torch.from_numpy(np.mean(aggregation_matrix, axis=0)).float().to(device)

    # Initialize Model
    model = SuperpixelNeuralNetwork(HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, Q, A).to(device)
    print(model)

    # Load training data
    train_dataset = ImageDataset(X_train, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #
    train_model(model, train_loader, criterion, optimizer)
