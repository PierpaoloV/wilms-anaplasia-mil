#!/usr/bin/env python3
"""
run_segmentation.py — Tissue/Background (TB) segmentation for a directory of WSIs.

Wraps the applynetwork_multiproc.py inference engine from the
pathology-segmentation-pipeline Docker image. Must be run inside that container.

Usage:
    python pipeline/01_segmentation/run_segmentation.py \
        --wsi_dir  /data/slides/ \
        --output_dir /data/masks/tb/ \
        [--models_dir /home/user/source/models] \
        [--gpu_count 1] \
        [--batch_size 90]
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# applynetwork_multiproc.py lives here inside the Docker image
INFERENCE_SCRIPT = Path("/home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py")
DEFAULT_MODELS_DIR = Path("/home/user/source/models")
TB_MODEL_SUBPATH   = "tb/playground_soft-cloud-137_best_model.pt"

WSI_EXTENSIONS = {".mrxs", ".svs", ".tif", ".tiff", ".ndpi"}


def download_tb_model(models_dir: Path) -> Path:
    """Download TB model weights if not already present."""
    sentinel = models_dir / TB_MODEL_SUBPATH
    if sentinel.exists():
        print(f"[tb] Model already present at {sentinel}")
        return sentinel
    print("[tb] Downloading model weights ...")
    script = Path(__file__).parent / "download_models.py"
    subprocess.run(
        [sys.executable, str(script), "tb"],
        env={**os.environ, "MODELS_DIR": str(models_dir)},
        check=True,
    )
    return sentinel


def run_tb_segmentation(wsi_path: Path, output_dir: Path, model_path: Path,
                        gpu_count: int, batch_size: int) -> None:
    """Run TB segmentation for a single WSI."""
    output_mask = output_dir / f"{wsi_path.stem}.tif"
    if output_mask.exists():
        print(f"  [skip] {wsi_path.name} — mask already exists")
        return

    print(f"  [seg] {wsi_path.name}")
    subprocess.run(
        [
            sys.executable, str(INFERENCE_SCRIPT),
            f"--input_wsi_path={wsi_path}",
            f"--output_wsi_path={output_mask}",
            f"--model_path={model_path}",
            "--read_spacing=4.0",
            "--write_spacing=4.0",
            "--tile_size=512",
            "--readers=20",
            "--writers=20",
            f"--batch_size={batch_size}",
            f"--gpu_count={gpu_count}",
            "--axes_order=cwh",
            "--custom_processor=torch_processor",
            "--reconstruction_information=[[0,0,0,0],[1,1],[96,96,96,96]]",
            "--quantize",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="TB segmentation for a directory of WSIs")
    parser.add_argument("--wsi_dir",    required=True,  help="Directory containing WSI files")
    parser.add_argument("--output_dir", required=True,  help="Directory for output TB masks")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR),
                        help="Directory where model weights are stored/downloaded")
    parser.add_argument("--gpu_count",  type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=90)
    args = parser.parse_args()

    wsi_dir    = Path(args.wsi_dir)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not INFERENCE_SCRIPT.exists():
        print(f"ERROR: inference script not found at {INFERENCE_SCRIPT}")
        print("Are you running inside the pathology-pipeline Docker container?")
        sys.exit(1)

    model_path = download_tb_model(models_dir)

    wsi_files = sorted(p for p in wsi_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS)
    if not wsi_files:
        print(f"No WSI files found in {wsi_dir}")
        sys.exit(1)

    print(f"Found {len(wsi_files)} WSIs — saving masks to {output_dir}\n")
    for wsi_path in wsi_files:
        run_tb_segmentation(wsi_path, output_dir, model_path, args.gpu_count, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
