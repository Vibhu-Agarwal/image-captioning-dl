import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, final_img_features: int):
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
        # to a smaller desired dimension (e.g., 512)
        self.feature_reducer = nn.Linear(resnet.fc.in_features, final_img_features)
        self.bn = nn.BatchNorm1d(final_img_features)

    def forward(self, images):
        # Disable gradient computation for the backbone
        with torch.no_grad():
            features = self.resnet(images)

        # Flatten the features from (batch_size, 2048, 1, 1) to (batch_size, 2048)
        features = features.view(features.size(0), -1)

        features = self.feature_reducer(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(
        self,
        img_feature_size: int,
        embedding_dim: int,
        hidden_size: int,
        vocab_size: int,
    ):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = SimpleRNN(
            img_feature_size=img_feature_size,
            input_size=embedding_dim,
            hidden_size=hidden_size,
            output_size=vocab_size,
        )

    def forward(self, img, captions):
        """
        Args:
            img: (batch_size, img_feature_size)
            captions: (batch_size, S)
        """

        # Cropping the last token (<end>)
        # If captions are: <start>, w1, w2, ..., <end> (Length S)
        # For input, we use: <start>, w1, w2, ..., wN (Length S-1)
        caption_inputs = captions[:, :-1]

        # shape: (batch_size, S-1, embedding_dim)
        captions_embeddings = self.embed(caption_inputs)

        # RNN runs on inputs (Length S-1)
        # outputs shape: (batch_size, S-1, vocab_size)
        outputs, _ = self.rnn(captions_embeddings, img)

        # The loss calculation in your training loop should use 'outputs' (Length S-1)
        # and 'captions[:, 1:]' (the targets: w1, w2, ..., <end>) (Length S-1)
        return outputs


class SimpleRNN(nn.Module):
    def __init__(
        self, input_size: int, img_feature_size: int, hidden_size: int, output_size: int
    ):
        super(SimpleRNN, self).__init__()

        self.i2h = nn.Linear(img_feature_size, hidden_size)
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.h2y = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        img_features: torch.Tensor | None = None,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: (batch_size, seq_len, input_size)
            h_prev: (batch_size, hidden_size)
            img_features: (batch_size, img_feature_size)
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
            if t == 0 and img_features is not None:
                h_t = self.tanh(
                    self.i2h(img_features) + self.x2h(x_t) + self.h2h(h_prev)
                )
            else:
                h_t = self.tanh(self.x2h(x_t) + self.h2h(h_prev))
            output = self.h2y(h_t)  # (batch_size, output_size)
            h_prev = h_t
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return outputs, h_t
