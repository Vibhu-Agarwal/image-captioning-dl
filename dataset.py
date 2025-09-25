import torch
from torchvision import transforms
from datasets import load_dataset, Dataset as HFDataset
from custom_types import CustomDataset
from language_utils import tokenize, create_vocabulary


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

    def vocabulary_size(self) -> int:
        if self._vocabulary is None:
            self._vocabulary = create_vocabulary(self)

        vocab, _ = self._vocabulary
        return len(vocab)

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


def get_train_dataset() -> Flickr8KDataset:
    ds = load_dataset("jxie/flickr8k")
    return Flickr8KDataset(hf_dataset=ds["train"], transform=transformations)
