import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--word-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=2881)
    args = parser.parse_args()

    manifest = pd.read_csv(args.source_manifest, encoding="utf-8")
    words = pd.read_csv(args.word_predictions, encoding="utf-8-sig")

    required_manifest = {"generated_image", "writer_id", "source_page_id"}
    required_words = {"filename", "ground_truth"}

    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Source manifest must contain: {sorted(required_manifest)}")

    if not required_words.issubset(words.columns):
        raise ValueError(f"Word predictions must contain: {sorted(required_words)}")

    manifest["filename"] = manifest["generated_image"].apply(
        lambda value: Path(str(value)).name
    )

    if manifest["filename"].duplicated().any():
        raise ValueError("Duplicate image filenames found in source manifest.")

    if words["filename"].duplicated().any():
        raise ValueError("Duplicate image filenames found in word predictions.")

    merged = words[["filename", "ground_truth"]].merge(
        manifest,
        on="filename",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    missing = merged["_merge"] != "both"

    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} word-only images could not be matched."
        )

    if len(merged) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} samples, found {len(merged)}."
        )

    result = merged.drop(
        columns=["filename", "ground_truth", "_merge"]
    ).copy()

    result["generated_image"] = result["generated_image"].apply(
        lambda value: Path(str(value)).name
    )

    if result["writer_id"].isna().any() or result["source_page_id"].isna().any():
        raise ValueError("Missing writer IDs or source page IDs found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Manifest: {output}")
    print(f"Samples: {len(result)}")
    print(f"Writers: {result['writer_id'].nunique()}")
    print(f"Source pages: {result['source_page_id'].nunique()}")
    print(f"Image paths: relative filenames")


if __name__ == "__main__":
    main()
