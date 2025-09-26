import os
import torch
from torchvision import transforms
from datasets import load_dataset, Dataset as HFDataset
from custom_types import CustomDataset
from language_utils import tokenize, create_vocabulary
from image_preprocess import get_transform


class Flickr8KDataset(CustomDataset):
    CAPTION_COUNT = 5
    VOCAB_CACHE_FILE = "flickr8k_vocab_cache.pt"

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

    def _load_or_create_vocabulary(self) -> tuple[dict[str, int], dict[int, str]]:
        if self._vocabulary is not None:
            return self._vocabulary

        if os.path.exists(self.VOCAB_CACHE_FILE):
            print(f"Loading vocabulary from cache: {self.VOCAB_CACHE_FILE}")
            try:
                vocab_data = torch.load(self.VOCAB_CACHE_FILE)
                self._vocabulary = vocab_data
                return vocab_data
            except Exception as e:
                print(f"Error loading vocabulary cache: {e}. Recreating vocabulary.")

        vocab_data = create_vocabulary(self)
        self._vocabulary = vocab_data

        try:
            torch.save(vocab_data, self.VOCAB_CACHE_FILE)
        except Exception as e:
            print(f"Warning: Could not save vocabulary cache: {e}")

        return vocab_data

    def tokenized_caption(self, idx: int) -> list[str]:
        image_idx, caption_idx = self._item_indices(idx)

        item = self.hf_dataset[image_idx]
        caption_text = str(item[f"caption_{caption_idx}"])

        return tokenize(caption_text)

    def vocabulary_size(self) -> int:
        if self._vocabulary is None:
            self._vocabulary = self._load_or_create_vocabulary()

        vocab, _ = self._vocabulary
        return len(vocab)

    def vocabularize_token(self, token: str) -> int:
        if self._vocabulary is None:
            self._vocabulary = self._load_or_create_vocabulary()

        vocab, _ = self._vocabulary
        return vocab.get(token, vocab["<unk>"])

    def inverse_vocabularize_token(self, index: int) -> str:
        if self._vocabulary is None:
            self._vocabulary = self._load_or_create_vocabulary()

        _, inv_vocab = self._vocabulary
        return inv_vocab.get(index, "<unk>")

    def _vocabularize_caption(self, caption: list[str]) -> list[int]:
        return [self.vocabularize_token(token) for token in caption]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        image_idx, _ = self._item_indices(idx)

        item = self.hf_dataset[image_idx]

        image = item["image"]
        if self.transform:
            image = self.transform(image)

        tokenized_caption = self.tokenized_caption(idx)
        vocabularized_caption = self._vocabularize_caption(tokenized_caption)

        return image, vocabularized_caption


def get_train_dataset() -> Flickr8KDataset:
    ds = load_dataset("jxie/flickr8k")
    transformations = get_transform()
    return Flickr8KDataset(hf_dataset=ds["train"], transform=transformations)
