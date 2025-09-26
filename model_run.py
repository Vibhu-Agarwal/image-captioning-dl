import torch
from model import CaptionGenerator
from torch.utils.tensorboard import SummaryWriter
from utils import (
    save_checkpoint,
    get_device,
)

FINAL_IMG_FEATURES = 512
WORD_EMBEDDING_DIM = 256
RNN_HIDDEN_LAYER_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100


def get_model(vocab_size: int) -> CaptionGenerator:
    device = get_device()
    model = CaptionGenerator(
        final_img_features=FINAL_IMG_FEATURES,
        embedding_dim=WORD_EMBEDDING_DIM,
        hidden_size=RNN_HIDDEN_LAYER_SIZE,
        vocab_size=vocab_size,
    ).to(device)
    return model


def get_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    return optimizer


def train_model(
    train_loader: torch.utils.data.DataLoader,
    model: CaptionGenerator,
    optimizer: torch.optim.Optimizer,
    start_epoch: int = 0,
    tf_experiment: str = "runs/my_image_captioning_experiment",
):
    device = get_device()
    writer = SummaryWriter(tf_experiment)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(start_epoch, NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            outputs = model(images, captions)

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
                    model,
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
            model,
            optimizer,
            avg_loss,
            filename=f"checkpoint_ep{epoch_num}.pth.tar",
        )

        writer.add_scalar("Epoch/Avg_Loss", avg_loss, epoch)

    print("Training finished.")
    writer.close()
