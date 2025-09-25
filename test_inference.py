import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from custom_types import CustomDataset
from PIL import Image
from typing import Callable

# Assume 'device' is defined globally or passed as a parameter if using this externally
# For this example, we'll assume it's available as a global from main.py
# If placing in a different file, you must pass 'device' as an argument.


def show_inference(
    train_dataset: CustomDataset,
    sample_index_in_hf_dataset: int,
    generate_caption: Callable[[torch.Tensor, int], list[str]],
    device: torch.device,
):
    """
    Retrieves an image, generates a caption, and displays the image,
    predicted caption, and all ground truth captions.

    Args:
        train_dataset (CustomDataset): The Flickr8KDataset instance.
        sample_index_in_hf_dataset (int): The index of the image in the base
                                          Hugging Face dataset (0 to 7999).
        generate_caption (callable): The function to generate the caption.
        device (torch.device): The device the tensors should be moved to.
    """
    CAPTION_COUNT = 5

    # 1. Get the original PIL Image and all 5 ground truth captions
    hf_item = train_dataset.hf_dataset[sample_index_in_hf_dataset]
    original_image_pil = hf_item["image"]

    ground_truths = []
    for i in range(CAPTION_COUNT):
        ground_truths.append(str(hf_item[f"caption_{i}"]))

    # 2. Get the pre-processed PyTorch Tensor for inference
    # Use the first caption's global index to get the image tensor
    global_index = sample_index_in_hf_dataset * CAPTION_COUNT
    image_tensor, _ = train_dataset[global_index]

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_tensor_batch = image_tensor.unsqueeze(0).to(device)

    # 3. Generate the caption
    predicted_tokens = generate_caption(image_tensor_batch, 50)
    predicted_caption = " ".join(predicted_tokens)

    # 4. Prepare Text for Visualization
    caption_text = f"Predicted: **{predicted_caption}**\n\nGround Truths:\n"
    for i, gt in enumerate(ground_truths):
        # Limit ground truth display length for clean visualization
        caption_text += f" • {gt[:80]}...\n"

    # 5. Display the results using Matplotlib
    plt.figure(figsize=(10, 6))

    # --- Image Subplot ---
    ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    ax1.imshow(original_image_pil)
    ax1.set_title(f"Image Index: {sample_index_in_hf_dataset}", fontsize=12)
    ax1.axis("off")

    # --- Text Subplot ---
    ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot

    # Display the captions as plain text in the second subplot
    ax2.text(
        0,
        1,
        caption_text.strip(),
        transform=ax2.transAxes,  # Coordinates relative to the axes (0,0 is bottom-left, 1,1 is top-right)
        fontsize=10,
        verticalalignment="top",
        wrap=True,
    )

    ax2.axis("off")  # Hide axes for the text box

    plt.tight_layout()
    plt.show()  # Opens the visualization window
