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
            nn.LeakyReLU(),
        )

        self.CNN_Branch = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # GCN Layers for Superpixels
        self.GCN_Branch = nn.Sequential(
            GCNLayer(128, 128, A),
            GCNLayer(128, 64, A),
        )

        # Fully Connected Layers for Classification
        self.fc_fusion = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Denoise Input
        noise = self.CNN_denoise(x)  # Shape: (batch_size, 128, height, width)
        clean_x = noise  # Direct connection for residual

        # Flatten Pixel Features
        clean_x_flatten = clean_x.permute(0, 2, 3, 1).reshape(batch_size, height * width,
                                                              -1)  # Shape: (batch_size, pixels, features)

        # Superpixel Features via Graph Convolution
        Q_batched = self.Q.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, superpixels, pixels)
        superpixels_flatten = torch.bmm(Q_batched, clean_x_flatten)  # Shape: (batch_size, superpixels, features)
        H = superpixels_flatten
        for gcn_layer in self.GCN_Branch:
            H = gcn_layer(H)  # Shape: (batch_size, superpixels, features)

        Q_T_batched = self.Q.T.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, pixels, superpixels)
        GCN_result = torch.bmm(Q_T_batched, H)  # Back to pixel level: Shape: (batch_size, pixels, features)

        # CNN Branch
        CNN_result = self.CNN_Branch(noise)  # Shape: (batch_size, 64, height, width)
        CNN_result = CNN_result.permute(0, 2, 3, 1).reshape(batch_size, height * width,
                                                            -1)  # Shape: (batch_size, pixels, features)

        # Debugging Shapes
        if GCN_result.shape[-1] != CNN_result.shape[-1]:
            print("Shape mismatch: GCN_result", GCN_result.shape, "CNN_result", CNN_result.shape)

        # Combine GCN and CNN Results

        combined_features = torch.cat([GCN_result, CNN_result],
                                      dim=-1)  # Correct combination: (batch_size, pixels, 192)
        print("GCN_result shape:", GCN_result.shape)  # Should be [batch_size, pixels, 64]
        print("CNN_result shape:", CNN_result.shape)


        #
        print("Combined features shape:", combined_features.shape)

        # Classification Layers
        combined_features = self.fc_fusion(combined_features)  # Shape: (batch_size, pixels, 128)
        combined_features = F.relu(combined_features)
        output = self.fc_out(combined_features)  # Shape: (batch_size, pixels, num_classes)
        output = F.softmax(output, dim=-1)
        output = output.mean(dim=1)

        return output
