from torchvision import transforms
from PIL import Image


def get_image(file_path: str) -> Image.Image:
    """
    Opens an image file and converts it to RGB format.
    """
    image = Image.open(file_path).convert("RGB")
    return image


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
