import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embedding_dim):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # From what I understand, children()'s order is the order of (nn.Module-type) variables assigned during the __init__ of the model
        # So, this does not guarantee the order of layers during the forward pass
        # But I've checked the order and it seems to be the same as the forward pass

        modules = list(resnet.children())

        # Remove the final classification layer
        modules = modules[:-1]

        self.resnet = nn.Sequential(*modules)

        # A fully connected layer to map the ResNet output (2048)
        # to a smaller, desired embedding dimension (e.g., 512)
        self.embed = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, images):
        # Disable gradient computation for the backbone
        with torch.no_grad():
            features = self.resnet(images)

        # Flatten the features from (batch_size, 2048, 1, 1) to (batch_size, 2048)
        features = features.view(features.size(0), -1)

        features = self.embed(features)
        features = self.bn(features)
        return features
