import torch
from torchvision import transforms
from datasets import load_dataset, Dataset as HFDataset
from custom_types import CustomDataset
from language_utils import tokenize, create_vocabulary
from typing import Iterable
from torch.nn.utils.rnn import pad_sequence


class Flickr8KDataset(CustomDataset):
    CAPTION_COUNT = 5

    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            hf_dataset (datasets.Dataset): The Hugging Face dataset split (e.g., ds['train']).
            transform (callable, optional): Optional transform to be applied on a sample image.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        self._vocabulary: tuple[dict[str, int], dict[int, str]] | None = None

    def __len__(self):
        return len(self.hf_dataset) * self.CAPTION_COUNT

    def _item_indices(self, idx: int) -> tuple[int, int]:
        image_idx = idx // self.CAPTION_COUNT
        caption_idx = idx % self.CAPTION_COUNT
        return image_idx, caption_idx

    def tokenized_caption(self, idx: int) -> list[str]:
        image_idx, caption_idx = self._item_indices(idx)

        item = self.hf_dataset[image_idx]
        caption_text = str(item[f"caption_{caption_idx}"])

        return tokenize(caption_text)

    def vocabularize(self, caption: list[str]) -> list[int]:
        if self._vocabulary is None:
            self._vocabulary = create_vocabulary(self)

        vocab, _ = self._vocabulary
        unk_token = vocab["<unk>"]

        return [vocab.get(word, unk_token) for word in caption]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        image_idx, _ = self._item_indices(idx)

        item = self.hf_dataset[image_idx]

        image = item["image"]
        if self.transform:
            image = self.transform(image)

        tokenized_caption = self.tokenized_caption(idx)
        vocabularized_caption = self.vocabularize(tokenized_caption)

        return image, vocabularized_caption


transformations = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


ds = load_dataset("jxie/flickr8k")
train_dataset = Flickr8KDataset(hf_dataset=ds["train"], transform=transformations)


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
