import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import get_train_dataset
from model import EncoderCNN, DecoderRNN
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, ensure_checkpoints_dir_exists

writer = SummaryWriter("runs/my_image_captioning_experiment")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

train_dataset = get_train_dataset()

FINAL_IMG_FEATURES = 512
WORD_EMBEDDING_DIM = 256
RNN_HIDDEN_LAYER_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100


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


def train_model(start_epoch: int = 0):
    encoder.train()
    decoder.train()

    for epoch in range(start_epoch, NUM_EPOCHS):
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

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch/Loss", loss.item(), step)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            if (batch_idx + 1) % 300 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                save_checkpoint(
                    epoch + 1,
                    encoder,
                    decoder,
                    optimizer,
                    avg_loss,
                    filename=f"checkpoint_ep{epoch+1}_step{batch_idx+1}.pth.tar",
                )

        avg_loss = total_loss / len(train_loader)
        epoch_num = epoch + 1
        print(
            f"Epoch [{epoch_num}/{NUM_EPOCHS}] completed. Average Loss: {avg_loss:.4f}"
        )

        save_checkpoint(
            epoch_num,
            encoder,
            decoder,
            optimizer,
            avg_loss,
            filename=f"checkpoint_ep{epoch_num}.pth.tar",
        )

        writer.add_scalar("Epoch/Avg_Loss", avg_loss, epoch)

    print("Training finished.")
    writer.close()


def generate_caption(image: torch.Tensor, max_len=50):
    """
    Args:
        image: Shape (1, 3, 224, 224).
        max_len: Maximum length of the generated caption.
    """

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    image = image.to(device)

    with torch.no_grad():
        features = encoder(image).unsqueeze(1)  # Shape: (1, 1, FINAL_IMG_FEATURES)

        start_token_idx = train_dataset.vocabularize_token("<start>")
        current_word = torch.tensor([[start_token_idx]]).long().to(device)

        # Initialize the generated sequence and states
        caption_indices = []
        hidden = None  # Start with default (zero) states

        for i in range(max_len):
            embeddings = decoder.embed(current_word)  # Shape: (1, 1, EMBED_SIZE)

            img_features = None
            if i == 0:
                img_features = features  # (1, 1, FINAL_IMG_FEATURES)

            outputs, hidden = decoder.rnn(
                embeddings, img_features, hidden
            )  # Shape: (1, 1, VOCAB_SIZE)
            outputs = outputs.squeeze(1)  # Shape: (1, VOCAB_SIZE)

            # Select the word with the highest probability (Greedy)
            _, predicted = outputs.max(1)  # predicted is a tensor of shape (1)
            predicted_idx = predicted.item()

            caption_indices.append(predicted_idx)

            if predicted_idx == train_dataset.vocabularize_token("<end>"):
                break

            current_word = predicted.unsqueeze(0)  # Shape: (1, 1)

        caption = [
            train_dataset.inverse_vocabularize_token(index) for index in caption_indices
        ]

        return caption[1:-1]  # Remove <start> and <end> tokens for final output


if __name__ == "__main__":
    ensure_checkpoints_dir_exists()
    checkpoint_file = "model_checkpoints/filename"

    start_epoch = 0
    if checkpoint_file:
        start_epoch = load_checkpoint(
            checkpoint_file, encoder, decoder, optimizer, device
        )

    train_model(start_epoch)
