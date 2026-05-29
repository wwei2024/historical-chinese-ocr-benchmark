from pathlib import Path
from PIL import Image, ImageOps


def resize_for_vlm(image_path: Path, output_path: Path, max_width: int = 1600) -> Path:
    """Resize image for VLM inference while preserving aspect ratio."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    if img.width > max_width:
        new_h = int(img.height * max_width / img.width)
        img = img.resize((max_width, new_h))
    img.save(output_path)
    return output_path


def crop_image(image_path: Path, output_path: Path, crop_box) -> Path:
    """Crop an image using PIL crop_box=(left, top, right, bottom)."""
    img = Image.open(image_path).convert('RGB')
    cropped = img.crop(crop_box)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
    return output_path


def crop_to_content(image_path: Path, output_path: Path, padding: int = 20, threshold: int = 245) -> Path:
    """Crop white margins from a scanned page."""
    img = Image.open(image_path).convert('L')
    mask = img.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        Image.open(image_path).convert('RGB').save(output_path)
        return Path(output_path)
    left, top, right, bottom = bbox
    left = max(left - padding, 0)
    top = max(top - padding, 0)
    right = min(right + padding, img.width)
    bottom = min(bottom + padding, img.height)
    return crop_image(image_path, output_path, (left, top, right, bottom))
