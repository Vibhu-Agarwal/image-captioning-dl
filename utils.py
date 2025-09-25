import os
import torch
import time


def ensure_checkpoints_dir_exists():
    """Ensure the directory for model checkpoints exists."""
    if not os.path.exists("model_checkpoints"):
        os.makedirs("model_checkpoints")


def save_checkpoint(
    epoch: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    filename="checkpoint.pth.tar",
):
    """Saves model and other training parameters at the current epoch."""
    print(f"Saving checkpoint to {filename}...")
    state = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    ts = time.strftime("%Y%m%d_%H%M%S_")
    filename = f"model_checkpoints/{ts}{filename}"
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


import torch
import os


def load_checkpoint(
    filename: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Loads a checkpoint and restores model, optimizer, and training state."""
    if not os.path.isfile(filename):
        print(
            f"Checkpoint file not found at '{filename}'. Starting training from epoch 1."
        )
        return 0  # Start from epoch 0

    print(f"Loading checkpoint from '{filename}'...")

    # Map location ensures the checkpoint loads correctly,
    # even if saved on a different device (e.g., GPU -> CPU/MPS)
    checkpoint = torch.load(filename, map_location=device)

    # 1. Load model states
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # 2. Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Get the starting epoch (we resume from the *next* epoch)
    start_epoch = checkpoint["epoch"]

    print(f"Checkpoint successfully loaded.")
    print(f"Resuming training from Epoch {start_epoch + 1}.")
    return start_epoch  # Return the epoch *completed*
