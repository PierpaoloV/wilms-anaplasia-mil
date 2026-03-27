#!/usr/bin/env python3
"""
prepare_csv.py — Generate the slide CSV required by slide2vec.

Scans a WSI directory and matches each slide with its TB segmentation mask,
producing a CSV with columns:
    sample_id, image_path, mask_path, spacing_at_level_0

Usage:
    python pipeline/02_feature_extraction/prepare_csv.py \
        --wsi_dir   /data/slides/ \
        --mask_dir  /data/masks/tb/ \
        --output_csv /data/slide2vec_input.csv \
        [--spacing 0.25]   # μm/px at level 0; read from slide if omitted
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

WSI_EXTENSIONS  = {".mrxs", ".svs", ".tif", ".tiff", ".ndpi"}
MASK_EXTENSIONS = {".tif", ".tiff"}


def get_spacing(wsi_path: Path) -> float | None:
    """Read level-0 spacing (μm/px) from WSI metadata via OpenSlide."""
    if not HAS_OPENSLIDE:
        return None
    try:
        slide = openslide.open_slide(str(wsi_path))
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        slide.close()
        return float(mpp_x) if mpp_x else None
    except Exception:
        return None


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    """Find a mask file matching the slide stem (any supported extension)."""
    for ext in MASK_EXTENSIONS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate slide2vec input CSV")
    parser.add_argument("--wsi_dir",    required=True,  help="Directory containing WSI files")
    parser.add_argument("--mask_dir",   required=True,  help="Directory containing TB mask TIFFs")
    parser.add_argument("--output_csv", required=True,  help="Path for output CSV")
    parser.add_argument("--spacing",    type=float, default=None,
                        help="Level-0 spacing in μm/px (read from slide metadata if omitted)")
    args = parser.parse_args()

    wsi_dir    = Path(args.wsi_dir)
    mask_dir   = Path(args.mask_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    wsi_files = sorted(p for p in wsi_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS)
    if not wsi_files:
        print(f"No WSI files found in {wsi_dir}")
        sys.exit(1)

    rows = []
    missing_masks = []

    for wsi_path in wsi_files:
        mask_path = find_mask(mask_dir, wsi_path.stem)
        if mask_path is None:
            missing_masks.append(wsi_path.name)
            continue

        spacing = args.spacing or get_spacing(wsi_path)
        rows.append({
            "sample_id":          wsi_path.stem,
            "image_path":         str(wsi_path.resolve()),
            "mask_path":          str(mask_path.resolve()),
            "spacing_at_level_0": spacing if spacing is not None else "",
        })

    if missing_masks:
        print(f"WARNING: no mask found for {len(missing_masks)} slides:")
        for name in missing_masks:
            print(f"  {name}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "mask_path", "spacing_at_level_0"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_csv}")
    if missing_masks:
        print(f"Skipped {len(missing_masks)} slides with no matching mask.")


if __name__ == "__main__":
    main()
