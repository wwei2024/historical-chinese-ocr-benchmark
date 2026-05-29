import base64
import time
from pathlib import Path
import requests

QWEN_VERTICAL_PROMPT = """This is a scanned page from an old Traditional Chinese book.

The text is written vertically in top-to-bottom columns ordered right-to-left.

Transcribe all visible text exactly as written.

Preserve the original reading order and line breaks.

Preserve uncommon or archaic characters exactly as they appear.
If uncertain, output the closest visible character rather than guessing.

Do not modernize characters.
Do not translate.
Do not summarize.
Do not add punctuation.
Do not explain the content."""


def run_qwen3_vl_ollama(
    image_path: Path,
    model_id: str = 'qwen3-vl:8b-instruct',
    prompt: str = QWEN_VERTICAL_PROMPT,
    max_tokens: int = 1024,
    verbose: bool = True,
) -> str:
    """Run Qwen3-VL through Ollama local HTTP API."""
    image_path = Path(image_path)
    if verbose:
        print(f'Using model: {model_id}')
        print(f'Image: {image_path}')
        print('Encoding image...')

    image_b64 = base64.b64encode(image_path.read_bytes()).decode('utf-8')
    payload = {
        'model': model_id,
        'prompt': prompt,
        'images': [image_b64],
        'stream': False,
        'options': {'temperature': 0.0, 'num_predict': max_tokens},
    }

    if verbose:
        print('Sending request to Ollama /api/generate...')
        start = time.time()

    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=900)

    if verbose:
        print(f'HTTP status: {response.status_code}')
        print(f'Completed in {time.time() - start:.2f} seconds')

    response.raise_for_status()
    text = response.json()['response']

    if verbose:
        print('\n--- Qwen OCR preview ---')
        print(text[:500])
    return text
