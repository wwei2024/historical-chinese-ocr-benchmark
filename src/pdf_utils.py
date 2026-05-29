from pathlib import Path
import fitz


def render_pdf_page_to_image(pdf_path: Path, output_path: Path, page_index: int = 0, dpi: int = 250) -> Path:
    """Render one PDF page to a PNG image."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(str(output_path))
    doc.close()
    return output_path
