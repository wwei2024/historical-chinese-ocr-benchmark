from pathlib import Path
import warnings
import time
import re
import numpy as np
from tqdm.auto import tqdm


def run_paddleocr(image_path: Path, verbose: bool = True):
    """Run PaddleOCR with a few robust initialization fallbacks."""
    from paddleocr import PaddleOCR
    init_errors = []
    ocr = None
    init_candidates = [
        dict(lang='ch', ocr_version='PP-OCRv5', use_angle_cls=True),
        dict(lang='ch', use_angle_cls=True),
        dict(lang='ch'),
    ]
    for kwargs in tqdm(init_candidates, desc='Initializing PaddleOCR', disable=not verbose):
        try:
            if verbose:
                print('Trying PaddleOCR init:', kwargs)
            ocr = PaddleOCR(**kwargs)
            break
        except Exception as e:
            init_errors.append((kwargs, repr(e)))
            ocr = None
    if ocr is None:
        raise RuntimeError(f'Failed to initialize PaddleOCR. Errors: {init_errors}')
    start = time.time()
    try:
        if verbose:
            print('Running PaddleOCR with predict()...')
        result = ocr.predict(str(image_path))
        api_used = 'predict'
    except Exception as e:
        if verbose:
            print('predict() failed, falling back to ocr.ocr(). Error:', repr(e))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = ocr.ocr(str(image_path))
        api_used = 'ocr'
    if verbose:
        print(f'PaddleOCR finished in {time.time() - start:.2f} seconds')
    return result, api_used


def extract_paddle_items(result):
    """Extract OCR items as dictionaries with text, box, and score."""
    items = []
    def add_item(text, box=None, score=None):
        text = str(text).strip()
        if text:
            items.append({'text': text, 'box': box, 'score': score})
    def walk(x):
        if x is None:
            return
        if hasattr(x, 'json'):
            try:
                return walk(x.json)
            except Exception:
                pass
        if hasattr(x, 'to_dict'):
            try:
                return walk(x.to_dict())
            except Exception:
                pass
        if isinstance(x, dict):
            rec_texts = x.get('rec_texts')
            rec_scores = x.get('rec_scores')
            rec_boxes = x.get('rec_boxes') or x.get('dt_polys') or x.get('rec_polys') or x.get('boxes')
            if isinstance(rec_texts, list):
                for i, text in enumerate(rec_texts):
                    score = rec_scores[i] if isinstance(rec_scores, list) and i < len(rec_scores) else None
                    box = rec_boxes[i] if isinstance(rec_boxes, list) and i < len(rec_boxes) else None
                    add_item(text, box, score)
                return
            for v in x.values():
                if isinstance(v, (dict, list, tuple)):
                    walk(v)
            return
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and isinstance(x[0], (list, tuple)) and isinstance(x[1], (list, tuple)) and len(x[1]) >= 1 and isinstance(x[1][0], str):
                text = x[1][0]
                score = x[1][1] if len(x[1]) > 1 else None
                box = x[0]
                add_item(text, box, score)
                return
            for item in x:
                walk(item)
    walk(result)
    return items


def box_center(box):
    """Return center x, center y from PaddleOCR box."""
    if box is None:
        return 0, 0
    try:
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            return (x1 + x2) / 2, (y1 + y2) / 2
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except Exception:
        return 0, 0


def is_probably_chinese_text(text):
    if not text:
        return False
    chinese_chars = re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text)
    return len(chinese_chars) >= max(1, len(text) * 0.4)


def filter_paddle_items(items, min_score: float = 0.35):
    filtered = []
    for item in items:
        text = str(item.get('text', '')).strip()
        score = item.get('score')
        if not text:
            continue
        if score is not None and score < min_score:
            continue
        if not is_probably_chinese_text(text):
            continue
        filtered.append(item)
    return filtered


def paddle_items_with_centers(result, min_score=0.35):
    items = extract_paddle_items(result)
    items = filter_paddle_items(items, min_score=min_score)
    enriched = []
    for item in items:
        cx, cy = box_center(item.get('box'))
        if cx == 0 and cy == 0:
            continue
        enriched.append({**item, 'cx': cx, 'cy': cy})
    return enriched


def group_items_into_vertical_columns(items, x_tolerance=28):
    """Group detected boxes into vertical columns, right-to-left."""
    if not items:
        return []
    items = sorted(items, key=lambda d: -d['cx'])
    columns = []
    for item in items:
        placed = False
        for col in columns:
            if abs(item['cx'] - col['mean_x']) <= x_tolerance:
                col['items'].append(item)
                col['mean_x'] = np.mean([i['cx'] for i in col['items']])
                placed = True
                break
        if not placed:
            columns.append({'mean_x': item['cx'], 'items': [item]})
    columns = sorted(columns, key=lambda c: -c['mean_x'])
    for col in columns:
        col['items'] = sorted(col['items'], key=lambda d: d['cy'])
    return columns


def text_from_vertical_columns(columns):
    return '\n'.join(''.join(item['text'] for item in col['items']) for col in columns)


def run_paddleocr_geometry_ordered(image_path: Path, min_score: float = 0.35, x_tolerance: int = 28, verbose: bool = True):
    result, api_used = run_paddleocr(image_path, verbose=verbose)
    items = paddle_items_with_centers(result, min_score=min_score)
    columns = group_items_into_vertical_columns(items, x_tolerance=x_tolerance)
    text = text_from_vertical_columns(columns)
    if verbose:
        print(f'PaddleOCR API used: {api_used}')
        print(f'Detected OCR items: {len(items)}')
        print(f'Detected columns: {len(columns)}')
        print(text[:500])
    return text, columns, items, result
