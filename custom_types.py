from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from torchvision import transforms
from abc import ABC, abstractmethod


class CustomDataset(Dataset, ABC):
    """
    Abstract base class for custom datasets.

    This class serves as a template for creating concrete dataset classes.
    It defines the essential methods that any dataset implementation should have.
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: transforms.Compose | None = None,
    ):
        """
        Initializes the custom dataset.

        Args:
            hf_dataset: A dataset from the Hugging Face datasets library.
            transform: A composition of torchvision transforms.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the processed image and its corresponding caption.
        """
        ...

    @abstractmethod
    def tokenized_caption(self, idx: int) -> list[str]:
        """
        Tokenizes the caption for a given dataset item.

        Args:
            idx: The index of the dataset item.

        Returns:
            A list of tokens representing the caption.
        """
        ...

    @abstractmethod
    def vocabularize_token(self, token: str) -> int:
        """
        Converts a token to its corresponding vocabulary index.

        Args:
            token: The token to convert.

        Returns:
            The index of the token in the vocabulary.
        """
        ...

    @abstractmethod
    def inverse_vocabularize_token(self, index: int) -> str:
        """
        Converts a vocabulary index back to its corresponding token.

        Args:
            index: The index to convert.

        Returns:
            The token corresponding to the index.
        """
        ...
