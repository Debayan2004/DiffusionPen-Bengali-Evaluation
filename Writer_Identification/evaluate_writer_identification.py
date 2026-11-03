import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from GRRNN import GrnnNet


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128


def load_writer_image(path):
    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        scale = min(IMAGE_HEIGHT / max(height, 1), IMAGE_WIDTH / max(width, 1))
        new_width = max(1, min(IMAGE_WIDTH, int(round(width * scale))))
        new_height = max(1, min(IMAGE_HEIGHT, int(round(height * scale))))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        array = 255.0 - np.asarray(image, dtype=np.float32)

    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    top = (IMAGE_HEIGHT - new_height) // 2
    left = (IMAGE_WIDTH - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = array
    canvas /= 255.0

    return torch.from_numpy(canvas).unsqueeze(0)


class SyntheticWriterDataset(Dataset):
    def __init__(self, dataframe, image_root):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(str(row["generated_image"]))

        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        image = load_writer_image(image_path)
        label = int(row["writer_id"])

        return image, torch.tensor(label, dtype=torch.long), str(index)


def calculate_classification_metrics(targets, predictions, num_classes):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)

    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    actual_count = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0
    )

    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count > 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0
    )

    present = actual_count > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    top1 = float((targets == predictions).mean())

    return top1, macro_f1, confusion


def evaluate_loader(model, loader, device, num_classes, output_dir):
    model.eval()

    targets = []
    predictions = []
    top5_correct = 0
    total = 0
    records = []

    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            k = min(5, num_classes)
            top_indices = logits.topk(k, dim=1).indices
            predicted = logits.argmax(dim=1)

            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())

            top5_correct += (
                top_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            )
            total += labels.size(0)

            for name, label, pred, top in zip(
                names,
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
                top_indices.cpu().tolist()
            ):
                records.append({
                    "image": name,
                    "ground_truth_writer_id": label,
                    "predicted_writer_id": pred,
                    "top5_writer_ids": " ".join(map(str, top))
                })

    top1, macro_f1, confusion = calculate_classification_metrics(
        targets, predictions, num_classes
    )

    top5 = top5_correct / max(1, total)

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        output_dir / "synthetic_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(confusion).to_csv(
        output_dir / "synthetic_confusion_matrix.csv",
        index=False
    )

    with open(output_dir / "synthetic_metrics.txt", "w", encoding="utf-8") as file:
        file.write(
            f"Samples: {total}\n"
            f"Top-1 accuracy: {top1:.8f}\n"
            f"Top-5 accuracy: {top5:.8f}\n"
            f"Macro-F1: {macro_f1:.8f}\n"
        )

    return top1, top5, macro_f1, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="../../Results/Writer_Identification")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / checkpoint_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    writer_to_id = checkpoint["writer_to_id"]
    num_classes = int(checkpoint["num_classes"])

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")

    required = {"source_page_id", "writer_id", "generated_image"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    manifest["source_page_id"] = manifest["source_page_id"].astype(str)
    manifest["writer_id"] = pd.to_numeric(
        manifest["writer_id"], errors="raise"
    ).astype(int)

    invalid_writers = sorted(
        set(manifest["source_page_id"]) - set(writer_to_id)
    )

    if invalid_writers:
        raise ValueError(
            f"Manifest contains writers absent from checkpoint mapping: "
            f"{invalid_writers[:10]}"
        )

    mapping_mismatches = manifest[
        manifest.apply(
            lambda row: writer_to_id[row["source_page_id"]] != row["writer_id"],
            axis=1
        )
    ]

    if len(mapping_mismatches):
        raise ValueError(
            f"{len(mapping_mismatches)} manifest writer IDs do not match "
            "the checkpoint writer mapping."
        )

    dataset = SyntheticWriterDataset(manifest, args.image_root)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )

    model = GrnnNet(
        1,
        num_classes=num_classes,
        mode=checkpoint.get("mode", "vertical")
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    top1, top5, macro_f1, total = evaluate_loader(
        model,
        loader,
        device,
        num_classes,
        output_dir
    )

    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    print("Synthetic images evaluated:", total)
    print("Writers:", manifest["writer_id"].nunique())
    print(f"Top-1 accuracy: {top1:.6f}")
    print(f"Top-5 accuracy: {top5:.6f}")
    print(f"Macro-F1: {macro_f1:.6f}")
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
