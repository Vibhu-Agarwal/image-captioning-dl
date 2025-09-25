import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import get_train_dataset

train_dataset = get_train_dataset()


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

for images, captions in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of captions shape: {captions.shape}")
    break
