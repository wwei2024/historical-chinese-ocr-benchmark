from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_cer_bar(results_df, output_path=None, width=0.35):
    x = np.arange(len(results_df))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, results_df['CER'], width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'])
    ax.set_ylabel('Character Error Rate (CER)')
    ax.set_title('OCR Comparison on Historical Traditional Chinese Page')
    for i, row in results_df.iterrows():
        ax.text(x[i], row['CER'], f"{row['CER']:.3f}", ha='center', va='bottom')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig, ax


def visualize_paddle_columns(image_path, columns, output_path=None, figsize=(8, 12)):
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_title('PaddleOCR boxes grouped into vertical columns')
    for col_idx, col in enumerate(columns, start=1):
        for item in col['items']:
            box = item.get('box')
            if box is None:
                continue
            try:
                if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                    x1, y1, x2, y2 = box
                else:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=1.2)
                ax.add_patch(rect)
                ax.text(x1, y1, str(col_idx), fontsize=8)
            except Exception:
                continue
    ax.axis('off')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig, ax
