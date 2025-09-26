import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from custom_types import CustomDataset
from PIL import Image
from image_preprocess import get_transform, get_image


def get_dataset_image_and_captions(
    train_dataset: CustomDataset, idx_in_hf_dataset: int
) -> tuple[torch.Tensor, list[str], Image.Image]:
    CAPTION_COUNT = 5

    # 1. Get the original PIL Image and all 5 ground truth captions
    hf_item = train_dataset.hf_dataset[idx_in_hf_dataset]
    original_image_pil = hf_item["image"]

    ground_truths = []
    for i in range(CAPTION_COUNT):
        ground_truths.append(str(hf_item[f"caption_{i}"]))

    # 2. Get the pre-processed PyTorch Tensor for inference
    # Use the first caption's global index to get the image tensor
    global_index = idx_in_hf_dataset * CAPTION_COUNT
    image_tensor, _ = train_dataset[global_index]
    return image_tensor, ground_truths, original_image_pil


def captions_text(ground_truths: list[str]) -> str:
    caption_text = "Ground Truths:\n"
    for i, gt in enumerate(ground_truths):
        caption_text += f"{i+1}. {gt[:80]}...\n"
    return caption_text


def preprocess_image_for_inference(
    image: Image.Image, device: torch.device
) -> torch.Tensor:
    """
    Preprocesses a PIL Image for model inference.
    Return Shape: (C, H, W)
    """
    transform = get_transform()
    image_tensor: torch.Tensor = transform(image)
    image_tensor = image_tensor.to(device)
    return image_tensor


def show_image_inference(
    image_path: str,
    generated_caption: list[str],
):
    image_pil = get_image(image_path)

    show_inference(
        title="Input Image",
        meta_text="",
        generated_caption=generated_caption,
        image_pil=image_pil,
    )


def show_dataset_inference(
    train_dataset: CustomDataset,
    sample_index_in_hf_dataset: int,
    generated_caption: list[str],
):
    _, ground_truths, original_image_pil = get_dataset_image_and_captions(
        train_dataset, sample_index_in_hf_dataset
    )

    caption_text = captions_text(ground_truths)

    title = f"Image Index: {sample_index_in_hf_dataset}"
    show_inference(
        title=title,
        meta_text=caption_text,
        generated_caption=generated_caption,
        image_pil=original_image_pil,
    )


def show_inference(
    image_pil: Image.Image,
    title: str,
    meta_text: str,
    generated_caption: list[str],
):
    """
    Retrieves an image, generates a caption, and displays the image,
    predicted caption, and all ground truth captions.
    """
    predicted_caption = " ".join(generated_caption)

    full_text = f"Predicted Caption:\n{predicted_caption}\n\n{meta_text}"

    # Display the results using Matplotlib
    plt.figure(figsize=(10, 6))

    # --- Image Subplot ---
    ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    ax1.imshow(image_pil)
    ax1.set_title(title, fontsize=12)
    ax1.axis("off")

    # --- Text Subplot ---
    ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot

    # Display the captions as plain text in the second subplot
    ax2.text(
        0,
        1,
        full_text.strip(),
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        wrap=True,
    )

    ax2.axis("off")

    plt.tight_layout()
    plt.show()
