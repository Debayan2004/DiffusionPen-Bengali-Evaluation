# Bengali Handwriting Evaluation

Evaluation of synthetic Bengali handwriting using HTRNet for text recognition and GR-RNN for writer identification.

## Dataset

3,025 synthetic handwritten samples from 303 writers. Dataset images and the original full manifest are not included.

## OCR Evaluation

Run: python OCR/evaluate_ocr.py --manifest PATH_TO_MANIFEST --image-root PATH_TO_IMAGES --batch-size 32

Single-word evaluation uses NFC normalization and includes exactly one whitespace-separated ground-truth token.

| Metric | Result |
|---|---:|
| Samples evaluated | 2,881 |
| Writers | 303 |
| Unique transcriptions | 667 |
| Correct words | 2,567 |
| Word accuracy | 89.10% |
| WER | 10.90% |
| CER | 4.65% |

## Writer Identification

Run: python Writer_Identification/evaluate_writer_identification.py --manifest Results/Writer_Identification/word_only_manifest.csv --image-root PATH_TO_IMAGES --checkpoint Writer_Identification/checkpoints/best_model.pt --output-dir Results/Writer_Identification --batch-size 16 --workers 0

| Metric | Result |
|---|---:|
| Top-1 accuracy | 53.73% |
| Top-5 accuracy | 73.20% |
| Macro-F1 | 59.47% |

## Installation

Install dependencies using: pip install -r requirements.txt


