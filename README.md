# Historical Chinese OCR Benchmark

A lightweight benchmark for comparing OCR systems on scanned historical Chinese books.

Current models:

- PaddleOCR
- Tesseract OCR
- Qwen3-VL via Ollama

The benchmark focuses on Traditional Chinese, vertical layout, historical printed books, and OCR quality comparison using Character Error Rate (CER).

## Motivation

Historical Chinese documents remain challenging for traditional OCR systems because of vertical page layout, historical typography, degraded scans, rare characters, and complex page structure.

Recent vision-language models such as Qwen3-VL can often outperform traditional OCR pipelines by jointly reasoning about layout and text. This project provides a simple framework for comparing these approaches.

## Project Structure

```text
data/
  samples/
  ground_truth/
notebooks/
src/
outputs/
examples/
```

## Installation

### Python environment

```bash
pip install -r requirements.txt
```

### Tesseract

macOS:

```bash
brew install tesseract
brew install tesseract-lang
```

Verify Traditional Chinese support:

```bash
tesseract --list-langs
```

Expected language pack:

```text
chi_tra
```

### PaddleOCR

PaddleOCR is installed through `requirements.txt`:

```bash
pip install paddleocr paddlepaddle
```

If you hit NumPy conflicts, use:

```bash
pip install "numpy<2.4"
```

## Qwen3-VL Setup with Ollama

Install Ollama:

```bash
brew install ollama
```

Start the server in one terminal:

```bash
ollama serve
```

Pull the model in another terminal:

```bash
ollama pull qwen3-vl:8b-instruct
```

Verify:

```bash
ollama list
```

## Qwen3-VL CLI Usage

This project recommends using Ollama for Qwen3-VL rather than direct MLX-VLM inference, because Ollama was more stable in local Mac testing.

Interactive CLI:

```bash
ollama run qwen3-vl:8b-instruct
```

Example OCR prompt:

```text
This is a scanned page from an old Traditional Chinese book.

The text is written vertically in top-to-bottom columns ordered right-to-left.

Transcribe all visible text exactly as written.

Preserve the original reading order and line breaks.

Preserve uncommon or archaic characters exactly as they appear.
If uncertain, output the closest visible character rather than guessing.

Do not modernize characters.
Do not translate.
Do not summarize.
Do not add punctuation.
Do not explain the content.
```

The Python runner in `src/qwen_runner.py` calls Ollama's local HTTP API.

## Metrics

Current metric:

- Character Error Rate (CER)

CER is computed after text normalization:

```text
CER = (Substitutions + Deletions + Insertions) / ReferenceLength
```

Lower is better. CER can be greater than 1.0 when the OCR output has many insertions and mismatches.

## Example Results

These are example values only. Results depend on scan quality and ground truth.

| Model | CER |
|---|---:|
| Qwen3-VL | 0.00 |
| PaddleOCR | 0.54 |
| Tesseract | 1.64 |

If Qwen output is used as ground truth, Qwen's CER should be 0.0 after identical normalization. For serious evaluation, use manually verified ground truth.

## Notebook

A placeholder notebook is included:

```text
notebooks/ocr_comparison.ipynb
```

It is intentionally minimal. The main reusable logic lives in `src/`.

## Future Work

- More historical book samples
- Additional OCR engines
- Layout-aware evaluation
- Reading-order accuracy metrics
- Batch benchmarking
- Human-verified ground-truth dataset creation
