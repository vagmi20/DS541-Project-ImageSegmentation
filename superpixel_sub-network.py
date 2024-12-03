import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, adjacency_matrix):
        super(GCNLayer, self).__init__()
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, features):
        # Compute normalized adjacency matrix
        D = torch.diag(torch.pow(self.adjacency_matrix.sum(1), -0.5))
        A_hat = D @ self.adjacency_matrix @ D
        # Apply GCN transformation
        features = A_hat @ features
        features = self.linear(features)
        return self.activation(features)


class SuperpixelNeuralNetwork(nn.Module):
    def __init__(self, height, width, channels, num_classes, Q, A):
        super(SuperpixelNeuralNetwork, self).__init__()
        
        # Input dimensions and class count
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.Q = Q
        self.A = A
        
        # Spectral-Spatial Convolutional Layers
        self.CNN_denoise = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.LeakyReLU()
        )
        
        self.CNN_Branch = nn.Sequential(
            SSConv(128, 128, kernel_size=5),
            SSConv(128, 128, kernel_size=5),
            SSConv(128, 64, kernel_size=5)
        )
        
        # GCN Layers for Superpixels
        self.GCN_Branch = nn.Sequential(
            GCNLayer(128, 128, A),
            GCNLayer(128, 64, A)
        )
        
        # Fully Connected Layers for Classification
        self.fc_fusion = nn.Linear(128 + 64, 128)
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Input shape: (Height x Width x Channels)
        h, w, c = x.shape
        
        # Denoise Input
        noise = self.CNN_denoise(torch.unsqueeze(x.permute(2, 0, 1), 0))  # Shape: (1, Channels, Height, Width)
        noise = torch.squeeze(noise, 0).permute(1, 2, 0)  # Shape: (Height, Width, Features)
        clean_x = noise  # Direct connection for residual
        
        # Flatten Pixel Features
        clean_x_flatten = clean_x.reshape([h * w, -1])  # Shape: (Pixels, Features)
        
        # Superpixel Features via Graph Convolution
        superpixels_flatten = torch.mm(self.Q.T, clean_x_flatten)  # Shape: (Superpixels, Features)
        H = superpixels_flatten
        for gcn_layer in self.GCN_Branch:
            H = gcn_layer(H)
        
        GCN_result = torch.mm(self.Q, H)  # Back to pixel level
        
        # CNN Branch
        CNN_result = self.CNN_Branch(torch.unsqueeze(clean_x.permute(2, 0, 1), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute(1, 2, 0).reshape([h * w, -1])  # Shape: (Pixels, Features)
        
        # Combine GCN and CNN Results
        combined_features = torch.cat([GCN_result, CNN_result], dim=-1)  # Shape: (Pixels, Combined Features)
        
        # Classification Layers
        combined_features = self.fc_fusion(combined_features)
        combined_features = F.relu(combined_features)
        output = self.fc_out(combined_features)
        output = F.softmax(output, dim=-1)
        
        return output


# Example Initialization
height, width, channels = 224, 224, 3  # Example dimensions
num_classes = 10  # Example number of classes

# Load precomputed superpixel matrices from slic.py output
Q = torch.from_numpy(np.load("aggregation_matrix.npy")).float().to(device)
A = torch.from_numpy(np.load("adjacency_matrix.npy")).float().to(device)

# Instantiate the model
model = SuperpixelNeuralNetwork(height, width, channels, num_classes, Q, A).to(device)

print(model)

