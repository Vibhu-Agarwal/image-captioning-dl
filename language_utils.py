from collections import Counter
from typing import Counter
from custom_types import CustomDataset
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")


def tokenize(sentence: str) -> list[str]:
    return ["<start>"] + word_tokenize(sentence.lower()) + ["<end>"]


def create_vocabulary(
    dataset: CustomDataset,
) -> tuple[dict[str, int], dict[int, str]]:

    word_counts: Counter[str] = Counter()

    for i in tqdm(range(len(dataset)), desc="Creating vocabulary"):
        caption = dataset.tokenized_caption(i)
        word_counts.update(caption)

    special_tokens: list[str] = ["<pad>", "<unk>", "<start>", "<end>"]
    vocab: dict[str, int] = {token: i for i, token in enumerate(special_tokens)}

    # Add words to the vocabulary based on their frequency
    for word, count in sorted(
        word_counts.items(), key=lambda item: item[1], reverse=True
    ):
        if word not in vocab:
            vocab[word] = len(vocab)

    vocab_reverse: dict[int, str] = {idx: word for word, idx in vocab.items()}

    return vocab, vocab_reverse
