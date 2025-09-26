import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import get_train_dataset
from model_run import get_model, get_optimizer, train_model
from utils import load_checkpoint, ensure_checkpoints_dir_exists, get_device


device = get_device()
print(f"Using device: {device}")

train_dataset = get_train_dataset()


def collate_fn(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images = torch.stack([item[0] for item in batch])
    all_captions = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    captions = pad_sequence(all_captions, batch_first=True, padding_value=0)
    return images, captions


def get_train_loader() -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )


model = get_model(train_dataset.vocabulary_size())
train_loader = get_train_loader()
optimizer = get_optimizer(model)


if __name__ == "__main__":
    ensure_checkpoints_dir_exists()
    checkpoint_file = "saved_checkpoints/single_layer_rnn_greedy_search_ep101.pth.tar"

    start_epoch = 0
    if checkpoint_file:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer, device)

    train_model(
        train_loader,
        model,
        optimizer,
        start_epoch=start_epoch,
        tf_experiment="runs/single_layer_rnn_greedy_search",
    )
