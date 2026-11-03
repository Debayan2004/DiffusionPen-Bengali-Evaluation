import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import OmegaConf

from models import HTRNet


OCR_DIR = Path(__file__).resolve().parent
REPO_DIR = OCR_DIR.parent


def normalize(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))

        previous = current

    return previous[-1]


def decode(sequence, i2c):
    collapsed = []

    for index in sequence:
        if not collapsed or index != collapsed[-1]:
            collapsed.append(index)

    return "".join(i2c[index] for index in collapsed if index != 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default=str(OCR_DIR / "checkpoints" / "final_htrnet_full2785.pt"))
    parser.add_argument("--config", default=str(OCR_DIR / "config.yaml"))
    parser.add_argument("--mapping", default=str(OCR_DIR / "htrnet_i2c.json"))
    parser.add_argument("--output-dir", default=str(REPO_DIR / "Results" / "OCR"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-samples", type=int, default=3025)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    image_root = Path(args.image_root)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    mapping_path = Path(args.mapping)
    output_dir = Path(args.output_dir)

    for path in [manifest_path, image_root, checkpoint_path, config_path, mapping_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, encoding="utf-8") as f:
        i2c = {int(k): v for k, v in json.load(f).items()}

    if len(i2c) != 66:
        raise ValueError(f"Expected 66 vocabulary characters, found {len(i2c)}")

    config = OmegaConf.load(config_path)
    config.device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    config.preproc.image_height = 64
    config.preproc.image_width = 256
    config.arch.head_type = "both"
    config.arch.rnn_type = "lstm"
    config.arch.rnn_layers = 3
    config.arch.rnn_hidden_size = 256
    config.arch.flattening = "maxpool"
    config.arch.stn = False

    model = HTRNet(config.arch, 67)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(config.device).eval()

    metadata = pd.read_csv(manifest_path, encoding="utf-8")
    required = {"generated_image", "writer_id", "transcription"}

    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing manifest columns: {required - set(metadata.columns)}")

    if metadata["generated_image"].duplicated().any():
        raise ValueError("Duplicate generated image paths found.")

    if args.expected_samples and len(metadata) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {len(metadata)}")

    results = []

    print("Checkpoint:", checkpoint_path)
    print("Device:", config.device)
    print("Samples:", len(metadata))
    print("Writers:", metadata["writer_id"].nunique())

    with torch.inference_mode():
        for start in range(0, len(metadata), args.batch_size):
            batch_metadata = metadata.iloc[start:start + args.batch_size]
            batch_images = []
            batch_records = []

            for row in batch_metadata.itertuples(index=False):
                image_path = Path(str(row.generated_image))

                if not image_path.is_absolute():
                    image_path = image_root / image_path.name

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                with Image.open(image_path) as image:
                    image = image.convert("L")

                    if image.size != (256, 64):
                        raise ValueError(f"Unexpected dimensions: {image_path}, {image.size}")

                    array = np.asarray(image, dtype=np.float32) / 255.0
                    array = 1.0 - array

                batch_images.append(array)
                batch_records.append({
                    "filename": image_path.name,
                    "writer_id": str(row.writer_id),
                    "ground_truth": str(row.transcription).strip()
                })

            batch = torch.from_numpy(np.stack(batch_images)).unsqueeze(1).float().to(config.device)
            output = model(batch)

            if isinstance(output, tuple):
                output = output[0]

            predictions = output.argmax(2).permute(1, 0).cpu().tolist()

            for record, predicted_ids in zip(batch_records, predictions):
                prediction = normalize(decode(predicted_ids, i2c))
                ground_truth = normalize(record["ground_truth"])
                errors = edit_distance(ground_truth, prediction)

                results.append({
                    "filename": record["filename"],
                    "writer_id": record["writer_id"],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "character_errors": errors,
                    "ground_truth_characters": len(ground_truth),
                    "cer": errors / max(len(ground_truth), 1),
                    "word_error": int(ground_truth != prediction)
                })

            print(f"Processed: {min(start + args.batch_size, len(metadata))}/{len(metadata)}")

    full_df = pd.DataFrame(results)
    full_df.to_csv(output_dir / "full_ocr_predictions.csv", index=False, encoding="utf-8-sig")

    full_df["word_count"] = full_df["ground_truth"].apply(lambda x: len(x.split()))
    word_df = full_df[full_df["word_count"] == 1].copy()
    excluded_df = full_df[full_df["word_count"] != 1].copy()

    word_df["exact_match"] = word_df["ground_truth"] == word_df["prediction"]
    word_df["word_error"] = (~word_df["exact_match"]).astype(int)
    word_df["character_errors"] = [
        edit_distance(gt, pred)
        for gt, pred in zip(word_df["ground_truth"], word_df["prediction"])
    ]
    word_df["ground_truth_characters"] = word_df["ground_truth"].str.len()
    word_df["cer"] = (
        word_df["character_errors"] /
        word_df["ground_truth_characters"].replace(0, 1)
    )

    total = len(word_df)
    correct = int(word_df["exact_match"].sum())
    incorrect = total - correct
    char_errors = int(word_df["character_errors"].sum())
    total_chars = int(word_df["ground_truth_characters"].sum())
    wer = incorrect / max(total, 1)
    cer = char_errors / max(total_chars, 1)

    per_writer = []

    for writer, group in word_df.groupby("writer_id"):
        samples = len(group)
        errors = int(group["character_errors"].sum())
        characters = int(group["ground_truth_characters"].sum())
        wrong = int(group["word_error"].sum())

        per_writer.append({
            "writer_id": writer,
            "samples": samples,
            "correct_words": samples - wrong,
            "incorrect_words": wrong,
            "word_accuracy": (samples - wrong) / samples,
            "wer": wrong / samples,
            "cer": errors / max(characters, 1)
        })

    writer_df = pd.DataFrame(per_writer).sort_values("cer", ascending=False)

    word_df.to_csv(output_dir / "word_only_predictions.csv", index=False, encoding="utf-8-sig")
    writer_df.to_csv(output_dir / "word_only_per_writer.csv", index=False, encoding="utf-8-sig")

    summary = {
        "original_samples": len(full_df),
        "included_single_word_samples": total,
        "excluded_multiword_samples": len(excluded_df),
        "included_writers": int(word_df["writer_id"].nunique()),
        "unique_word_transcriptions": int(word_df["ground_truth"].nunique()),
        "correct_words": correct,
        "incorrect_words": incorrect,
        "word_accuracy": correct / max(total, 1),
        "wer": wer,
        "character_errors": char_errors,
        "total_characters": total_chars,
        "cer": cer,
        "unicode_normalization": "NFC",
        "filter": "Exactly one whitespace-separated ground-truth token"
    }

    with open(output_dir / "word_only_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nWORD-ONLY HTRNET EVALUATION")
    print("Original samples:", len(full_df))
    print("Included single-word samples:", total)
    print("Excluded multiword samples:", len(excluded_df))
    print("Included writers:", word_df["writer_id"].nunique())
    print("Unique word transcriptions:", word_df["ground_truth"].nunique())
    print("Correct words:", correct)
    print("Incorrect words:", incorrect)
    print("Word accuracy:", f"{100 * (1 - wer):.2f}%")
    print("WER:", f"{wer:.6f}")
    print("Character errors:", char_errors)
    print("Total characters:", total_chars)
    print("CER:", f"{cer:.6f}")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
