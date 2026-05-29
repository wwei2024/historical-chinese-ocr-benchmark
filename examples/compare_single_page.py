from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pdf_utils import render_pdf_page_to_image
from src.preprocess import resize_for_vlm
from src.paddle_runner import run_paddleocr_geometry_ordered
from src.tesseract_runner import run_tesseract_ocr
from src.qwen_runner import run_qwen3_vl_ollama, QWEN_VERTICAL_PROMPT
from src.metrics import normalize_text_for_cer, cer
from src.visualization import plot_cer_bar


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or '', encoding='utf-8')


def main():
    pdf_path = ROOT / 'data' / 'samples' / 'page_001.pdf'
    gt_path = ROOT / 'data' / 'ground_truth' / 'page_001.txt'
    work_dir = ROOT / 'outputs'
    rendered_path = work_dir / 'page_001_rendered.png'
    vlm_path = work_dir / 'page_001_vlm_w1600.png'

    render_pdf_page_to_image(pdf_path, rendered_path, dpi=250)
    resize_for_vlm(rendered_path, vlm_path, max_width=1600)

    qwen_text = run_qwen3_vl_ollama(vlm_path, model_id='qwen3-vl:8b-instruct', prompt=QWEN_VERTICAL_PROMPT, max_tokens=1024)
    save_text(work_dir / 'qwen' / 'page_001.txt', qwen_text)

    paddle_text, _, _, _ = run_paddleocr_geometry_ordered(rendered_path, min_score=0.35, x_tolerance=28)
    save_text(work_dir / 'paddle' / 'page_001.txt', paddle_text)

    tesseract_text = run_tesseract_ocr(rendered_path, lang='chi_tra', psm=5)
    save_text(work_dir / 'tesseract' / 'page_001_psm5.txt', tesseract_text)

    ground_truth = gt_path.read_text(encoding='utf-8').strip()
    if not ground_truth or ground_truth.startswith('#'):
        print('No manually verified ground truth found. Using Qwen output as temporary reference.')
        ground_truth = qwen_text

    gt_norm = normalize_text_for_cer(ground_truth)
    rows = []
    for model_name, output in [('Tesseract', tesseract_text), ('PaddleOCR', paddle_text), ('Qwen3-VL', qwen_text)]:
        output_norm = normalize_text_for_cer(output)
        rows.append({
            'model': model_name,
            'raw_output_chars': len(output or ''),
            'normalized_output_chars': len(output_norm),
            'CER': cer(gt_norm, output_norm),
        })

    results_df = pd.DataFrame(rows)
    results_csv = work_dir / 'ocr_comparison_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(results_df)
    plot_cer_bar(results_df, output_path=work_dir / 'figures' / 'cer_comparison.png')


if __name__ == '__main__':
    main()
