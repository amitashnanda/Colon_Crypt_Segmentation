#!/usr/bin/env bash
# =============================================================================================
# Colon Crypt Segmentation (PyTorch) - Validation: TF Downstream Steps on a TF Re-run
#
# This script runs the TF workflow downstream on a TF inference output, with the TF scripts of
# ../src/inference:
#   u_shape_split.py -> ushape_bottom_up.py -> color_detection.py -> per-slide table -> p-values
# It is used to re-run TF end to end on this machine (tf_reference_inference.py writes ROOT/inf) and so
# to measure how much TF itself varies from the original TF results (the noise floor for the PyTorch
# comparison).
#
# Usage (from anywhere; ROOT, TF_PY and TORCH_PY can be overridden):
#   bash src/validation/run_tf_reference_downstream.sh
#
# Author: Amitash Nanda
# Date: 2026-10-09
# =============================================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # pytorch_mrcnn/
TF_PY="${TF_PY:-/pscratch/sd/a/ananda/conda_envs/cdx2_tf1_ref/bin/python}"
TORCH_PY="${TORCH_PY:-/pscratch/sd/a/ananda/conda_envs/torch_env/bin/python}"
ROOT="${ROOT:-validation/tf_reference}"
CDX=dataset/Whole_Slides_Segments_Processed/CDX2_Data
TF_SCRIPTS=../src/inference
PROTOCOLS=(Boiling-2min Boiling-5min Boiling-15min Pressure-3min)

echo "== U-shape split (src/inference/u_shape_split.py of the TF code)"
"$TF_PY" -B "$TF_SCRIPTS/u_shape_split.py" --inf_root "$ROOT/inf" --out_root "$ROOT/us_split" > "$ROOT/us_split.log" 2>&1

echo "== U-shape extraction (TF src/inference/ushape_bottom_up.py, imgaug), one process per protocol"
pids=()
for p in "${PROTOCOLS[@]}"; do
    "$TF_PY" -B "$TF_SCRIPTS/ushape_bottom_up.py" --us_split_root "$ROOT/us_split/$p" --original_root "$CDX/$p" \
        --output_root "$ROOT/us_final/$p" > "$ROOT/us_final_$p.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "== colour detection (TF src/inference/color_detection.py), one process per protocol"
pids=()
for p in "${PROTOCOLS[@]}"; do
    "$TF_PY" -B "$TF_SCRIPTS/color_detection.py" --src "$ROOT/us_final/$p" --dest "$ROOT/final/$p" > "$ROOT/final_$p.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "== per-slide table and p-values"
"$TORCH_PY" -B src/analysis/build_res_table.py --final_root "$ROOT/final" --out "$ROOT/res_tf_rerun.txt"
"$TORCH_PY" -B src/analysis/plot_pvalues.py --res "$ROOT/res_tf_rerun.txt" --out "$ROOT/plots/human_tf_rerun.png"
