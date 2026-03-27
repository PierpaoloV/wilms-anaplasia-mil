#!/bin/bash
#SBATCH --job-name=wilms-pipeline
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# --------------------------------------------------------------------
# Wilms Tumour Anaplasia Classification — full pipeline
#
# Stages:
#   01  TB segmentation      (applynetwork_multiproc via Docker)
#   02  Prepare slide2vec CSV
#   03  Feature extraction   (slide2vec)
#   04  AMIL training
#   05  Inference + heatmaps
#
# Edit the PATHS section below before submitting.
# --------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# --------------------------------------------------------------------
# PATHS — edit these
# --------------------------------------------------------------------
WSI_DIR=/data/slides
MASK_DIR=/data/masks/tb
SLIDE2VEC_CSV=/data/slide2vec_input.csv
FEATURES_DIR=/data/features
EXPERIMENTS_DIR=/home/user/experiments_complete

CONFIG_AMIL=configs/runs.yaml
CONFIG_S2V=configs/slide2vec.yaml

DOCKER_IMAGE=pierpaolov93/pathology-pipeline:cellvit
# --------------------------------------------------------------------

echo "========================================"
echo "Stage 01: TB Segmentation"
echo "========================================"
docker run --rm --gpus all \
    -v "${WSI_DIR}":"${WSI_DIR}" \
    -v "${MASK_DIR}":"${MASK_DIR}" \
    -v "$(pwd)":/workspace \
    "${DOCKER_IMAGE}" \
    python /workspace/pipeline/01_segmentation/run_segmentation.py \
        --wsi_dir "${WSI_DIR}" \
        --output_dir "${MASK_DIR}" \
        --gpu_count 1

echo "========================================"
echo "Stage 02: Prepare slide2vec CSV"
echo "========================================"
python pipeline/02_feature_extraction/prepare_csv.py \
    --wsi_dir   "${WSI_DIR}" \
    --mask_dir  "${MASK_DIR}" \
    --output_csv "${SLIDE2VEC_CSV}"

echo "========================================"
echo "Stage 03: Feature extraction (slide2vec)"
echo "========================================"
docker run --rm --gpus all \
    -v "${WSI_DIR}":"${WSI_DIR}" \
    -v "${MASK_DIR}":"${MASK_DIR}" \
    -v "${FEATURES_DIR}":"${FEATURES_DIR}" \
    -v "$(pwd)":/workspace \
    "${DOCKER_IMAGE}" \
    python -m slide2vec --config /workspace/"${CONFIG_S2V}"

echo "========================================"
echo "Stage 04: AMIL Training"
echo "========================================"
docker run --rm --gpus all \
    -v "${FEATURES_DIR}":"${FEATURES_DIR}" \
    -v "${EXPERIMENTS_DIR}":"${EXPERIMENTS_DIR}" \
    -v "$(pwd)":/workspace \
    "${DOCKER_IMAGE}" \
    python /workspace/pipeline/03_classification/mil_main.py \
        --config /workspace/"${CONFIG_AMIL}" \
        --run all

echo "========================================"
echo "Stage 05: Inference + Heatmaps"
echo "========================================"
docker run --rm --gpus all \
    -v "${FEATURES_DIR}":"${FEATURES_DIR}" \
    -v "${EXPERIMENTS_DIR}":"${EXPERIMENTS_DIR}" \
    -v "$(pwd)":/workspace \
    "${DOCKER_IMAGE}" \
    python /workspace/pipeline/03_classification/mil_inference.py \
        --config /workspace/"${CONFIG_AMIL}" \
        --run all \
        --device cuda \
        --extract_region \
        --combine_subplots

echo "========================================"
echo "Pipeline complete."
echo "========================================"
