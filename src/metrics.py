import re
from rapidfuzz.distance import Levenshtein


def normalize_text_for_cer(text: str) -> str:
    """Conservative normalization for Traditional Chinese OCR evaluation."""
    if text is None:
        return ''
    s = str(text)
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[，。、；：「」『』（）()\[\]【】《》〈〉,.!?！？:;\-—_]', '', s)
    return s


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(reference, hypothesis) / len(reference)."""
    ref = reference or ''
    hyp = hypothesis or ''
    if len(ref) == 0:
        return float('nan')
    return Levenshtein.distance(ref, hyp) / len(ref)
