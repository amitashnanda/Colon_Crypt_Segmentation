#!/usr/bin/env bash
# =============================================================================================
# Colon Crypt Segmentation (PyTorch) - Validation: Run All TF-vs-PyTorch Checks
#
# This script re-runs every TF-vs-PyTorch check and comparison. All outputs go to validation/compare/.
#
# Inputs: the TF results (../inference), the PyTorch results (inference/, from
# src/inference/run_inference_pipeline.py with --stages_dir validation/torch_stages), the TF re-run
# (validation/tf_reference, from tf_reference_inference.py + run_tf_reference_downstream.sh), the April
# run (validation/april_pytorch_run) and the downstream-only run (validation/torch_downstream_on_tf_masks,
# from run_downstream_on_tf_masks.sh).
#
# Usage (from anywhere; TF_PY and TORCH_PY can be overridden):
#   bash src/validation/run_checks.sh
#
# Author: Amitash Nanda
# Date: 2026-10-09
# =============================================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # pytorch_mrcnn/
TF_PY="${TF_PY:-/pscratch/sd/a/ananda/conda_envs/cdx2_tf1_ref/bin/python}"
TORCH_PY="${TORCH_PY:-/pscratch/sd/a/ananda/conda_envs/torch_env/bin/python}"
TF=../inference
CDX=dataset/Whole_Slides_Segments_Processed/CDX2_Data
V=validation
C=$V/compare
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
mkdir -p "$C"

echo "== weights"
"$TORCH_PY" -B src/weight_conversion/check_weight_conversion.py --h5 models/notransfer.h5 --pth models/notransfer_torch.pth \
    2>/dev/null > "$C/check_weight_conversion.txt"
echo "== preprocessing (imgaug in the TF env, replica in torch_env)"
"$TF_PY" -B src/validation/check_preprocessing_parity.py --mode imgaug --src "$CDX" --out "$C/preprocess_tf_imgaug.json" > "$C/preprocess_tf.log" 2>&1
"$TORCH_PY" -B src/validation/check_preprocessing_parity.py --mode replica --src "$CDX" --out "$C/preprocess_torch_replica.json" > "$C/preprocess_torch.log" 2>&1
"$TORCH_PY" -B src/validation/check_preprocessing_parity.py --compare "$C/preprocess_tf_imgaug.json" "$C/preprocess_torch_replica.json" > "$C/check_preprocessing.txt"
echo "== split"
"$TORCH_PY" -B src/validation/check_split_parity.py --root "$TF" > "$C/check_split.txt"
"$TORCH_PY" -B src/validation/check_split_parity.py --root "$TF" --no_compat > "$C/check_split_nocompat.txt"
echo "== patch frame"
"$TORCH_PY" -B src/validation/check_patch_frame_shift.py --root "$TF" > "$C/check_patch_frame_shift.txt"
echo "== stages, PyTorch vs TF re-run"
"$TORCH_PY" -B src/validation/compare_stages_with_tf.py --tf "$V/tf_reference/stages" --torch "$V/torch_stages" \
    --json "$C/stages_torch_vs_tf_rerun.json" > "$C/stages_torch_vs_tf_rerun.txt"
grep -v -e "^  " -e "^tiles compared" -e "^wrote" "$C/stages_torch_vs_tf_rerun.txt" > "$C/stages_torch_vs_tf_rerun_summary.txt"
echo "== result folders against the original TF results"
for run in "torch_parity:TF:inference:PyTorch" "tf_rerun:TF_original:$V/tf_reference:TF_rerun" \
           "torch_april:TF_original:$V/april_pytorch_run:PyTorch_April"; do
    IFS=: read -r tag a_name root b_name <<< "$run"
    "$TORCH_PY" -B src/validation/compare_tf_pytorch_results.py --a "$TF" --a_name "$a_name" --b "$root" --b_name "$b_name" \
        --json "$C/tf_original_vs_$tag.json" | grep -v "^wrote " > "$C/tf_original_vs_$tag.txt"
done
