import os
import torch
import time


def ensure_checkpoints_dir_exists():
    """Ensure the directory for model checkpoints exists."""
    if not os.path.exists("model_checkpoints"):
        os.makedirs("model_checkpoints")


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    filename="checkpoint.pth.tar",
):
    """Saves model and other training parameters at the current epoch."""
    print(f"Saving checkpoint to {filename}...")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    ts = time.strftime("%Y%m%d_%H%M%S_")
    filename = f"model_checkpoints/{ts}{filename}"
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
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

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]

    print(f"Checkpoint epoch:{start_epoch} successfully loaded.")
    return start_epoch


def get_device() -> torch.device:
    """Get the available device: GPU, MPS, or CPU."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device
