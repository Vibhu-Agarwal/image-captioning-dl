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


class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = SimpleRNN(
            input_size=embedding_dim, hidden_size=hidden_size, output_size=vocab_size
        )

    def forward(self, features, captions):
        embeddings = self.embed(captions)

        # 2. Concatenate the image features as the *first* time step input
        # Features shape: (batch_size, embedding_dim)
        # Embeddings shape: (batch_size, seq_len, embedding_dim)
        # Concatenated input shape: (batch_size, seq_len + 1, embedding_dim)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        outputs, _ = self.rnn(inputs)

        return outputs


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.h2y = nn.Linear(hidden_size, output_size)

    def forward(
        self, input: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: (batch_size, seq_len, input_size)
            output: (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = input.size()

        if h_prev is None:
            hidden_size = self.h2h.in_features
            h_prev = torch.zeros(
                batch_size, hidden_size, device=input.device, dtype=input.dtype
            )

        outputs = []
        h_t = h_prev
        for t in range(seq_len):
            x_t = input[:, t, :]
            h_t = self.tanh(self.i2h(x_t) + self.h2h(h_prev))
            output = self.h2y(h_t)  # (batch_size, output_size)
            h_prev = h_t
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return outputs, h_t
