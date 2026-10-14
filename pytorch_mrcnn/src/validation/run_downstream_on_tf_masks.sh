#!/usr/bin/env bash
# =============================================================================================
# Colon Crypt Segmentation (PyTorch) - Validation: Downstream Steps on TF's Masks
#
# This script runs the PyTorch downstream (src/inference/ushape_bottom_up.py + color_detection.py) on
# TF's ORIGINAL crypt masks (TF us_split), then compares the patches, per-slide values and p-values with
# TF's original us_final/ and final/. Any difference here comes from the downstream code, not the model.
#
# Usage (torch_env, from anywhere; TF_RESULTS and OUT can be overridden):
#   bash src/validation/run_downstream_on_tf_masks.sh
#
# Author: Amitash Nanda
# Date: 2026-10-09
# =============================================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # pytorch_mrcnn/
TF="${TF_RESULTS:-../inference}"
CDX=dataset/Whole_Slides_Segments_Processed/CDX2_Data
OUT="${OUT:-validation/torch_downstream_on_tf_masks}"
PROTOCOLS=(Boiling-2min Boiling-5min Boiling-15min Pressure-3min)
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" validation/compare
# TF's masks and split, linked so that the comparison sees the same inf/ and us_split/ in both roots
ln -sfn "$(realpath --relative-to="$OUT" "$TF/inf")" "$OUT/inf"
ln -sfn "$(realpath --relative-to="$OUT" "$TF/us_split")" "$OUT/us_split"
pids=()
for p in "${PROTOCOLS[@]}"; do
    python src/inference/ushape_bottom_up.py --us_split_root "$TF/us_split/$p" \
        --original_root "$CDX/$p" --output_root "$OUT/us_final/$p" > "$OUT/us_final_$p.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
grep -h "^\[check\]" "$OUT"/us_final_*.log
pids=()
for p in "${PROTOCOLS[@]}"; do
    python src/inference/color_detection.py --src "$OUT/us_final/$p" --dest "$OUT/final/$p" > "$OUT/final_$p.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python src/analysis/build_res_table.py --final_root "$OUT/final" --out "$OUT/res_downstream_on_tf_masks.txt"
python src/validation/compare_tf_pytorch_results.py --a "$TF" --a_name TF --b "$OUT" --b_name PyTorch_downstream \
    --json validation/compare/tf_original_vs_torch_downstream_on_tf_masks.json
