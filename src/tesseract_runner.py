from pathlib import Path
from PIL import Image
import pytesseract
import time


def run_tesseract_ocr(image_path: Path, lang: str = 'chi_tra', psm: int = 5, oem: int = 1, verbose: bool = True) -> str:
    """Run Tesseract OCR on a Traditional Chinese scanned page."""
    image_path = Path(image_path)
    if verbose:
        print(f'Running Tesseract on: {image_path}')
        print(f'Language: {lang}; PSM: {psm}; OEM: {oem}')
    img = Image.open(image_path).convert('RGB')
    config = f'--oem {oem} --psm {psm}'
    start = time.time()
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    if verbose:
        print(f'Tesseract finished in {time.time() - start:.2f} seconds')
        print(text[:500])
    return text
