import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import get_train_dataset
from model import EncoderCNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = get_train_dataset()

FINAL_IMG_FEATURES = 512
WORD_EMBEDDING_DIM = 256
RNN_HIDDEN_LAYER_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5


def collate_fn(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images = torch.stack([item[0] for item in batch])
    all_captions = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    captions = pad_sequence(all_captions, batch_first=True, padding_value=0)
    return images, captions


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

encoder = EncoderCNN(final_img_features=FINAL_IMG_FEATURES).to(device)
decoder = DecoderRNN(
    img_feature_size=FINAL_IMG_FEATURES,
    embedding_dim=WORD_EMBEDDING_DIM,
    hidden_size=RNN_HIDDEN_LAYER_SIZE,
    vocab_size=train_dataset.vocabulary_size(),
).to(device)

criterion = torch.nn.CrossEntropyLoss()

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)


def train_model():
    encoder.train()
    decoder.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            img_features = encoder(images)
            outputs = decoder(img_features, captions)

            # Reshape outputs and targets for loss calculation
            batch_size, seq_len, vocab_size = outputs.size()
            outputs = outputs.view(batch_size * seq_len, vocab_size)
            targets = captions[:, 1:].flatten()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss: {avg_loss:.4f}")

    print("Training finished.")


def basic_print_shapes():
    print("Printing shapes of batches and model outputs for verification...")
    print(f"Vocabulary size: {train_dataset.vocabulary_size()}")

    for images, captions in train_loader:
        images = images.to(device)
        captions = captions.to(device)
        print(f"Batch of images shape: {images.shape} | Device: {images.device}")
        print(f"Batch of captions shape: {captions.shape} | Device: {captions.device}")

        img_features = encoder(images)
        print(
            f"Image features shape: {img_features.shape} | Device: {img_features.device}"
        )
        outputs = decoder(img_features, captions)
        print(f"Outputs shape: {outputs.shape} | Device: {outputs.device}")
        break


train_model()
