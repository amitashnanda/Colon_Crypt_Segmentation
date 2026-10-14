"""
Colon Crypt Segmentation - Whole TF Inference Pipeline in One Command

This script runs the whole TF inference pipeline on tiles that are already colour-normalised and
Hough-aligned (dataset/Whole_Slides_Segments_Processed/CDX2_Data, or the output of the two
src/preprocess/ scripts). It calls the step scripts of this folder in order and writes the same folders
as inference/:

    1 inference  my_inference.py      -> <dest>/inf/<protocol>/<slide>/{predicted_images,predicted_mask}/
    2 split      u_shape_split.py     -> <dest>/us_split/<protocol>/<slide>/{accepted,fragmented,rejected}/
    3 ushape     ushape_bottom_up.py  -> <dest>/us_final/<protocol>/<slide>/{accepted,fragmented}/
    4 color      color_detection.py   -> <dest>/final/<protocol>/<slide>/final.txt (+ mask_and/, blue_brown_mask/, pdfs)
    5 table      one row per slide    -> <dest>/res_new.txt (Mouse-DSS left out); the Welch p-values are printed

Steps 3 and 4 run one process per protocol; the output of every step goes to <dest>/logs/.
--start <step> reuses the folder of the step before it from <dest>. Existing results are never replaced
unless --overwrite is given.

Usage (TF environment, environment_cdx2_tf1_ref.yml, from the repository root: relative paths, including the
defaults of --src and --model, are taken from the working directory; the steps then run from the repository root):
    python src/inference/run_inference_pipeline.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \\
        --dest inference_run --model models/notransfer.h5

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


# --- Default Settings ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STEPS = ["inference", "split", "ushape", "color", "table"]
STAGE_DIR = {"inference": "inf", "split": "us_split", "ushape": "us_final", "color": "final"}
PROTOCOLS = ["Boiling-2min", "Boiling-5min", "Boiling-15min", "Pressure-3min"]   # row order of the table
BOILING = PROTOCOLS[:3]


def run_jobs(jobs, n_parallel, env):
    """
    Run step commands from the repository root, at most n_parallel at a time; stop everything if one fails.

    Args:
        jobs: list of (command, log file) pairs
        n_parallel: maximum number of processes running at once
        env: environment of the processes
    """
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < n_parallel:
            cmd, log_path = pending.pop(0)
            log = open(log_path, "w")
            running.append((subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        time.sleep(1)
        for job in list(running):
            proc, log, log_path = job
            if proc.poll() is None:
                continue
            log.close()
            running.remove(job)
            if proc.returncode != 0:
                for other, _, _ in running:
                    other.kill()
                sys.exit("step failed (exit code %d), see %s" % (proc.returncode, log_path))


def subfolders(path):
    """
    Args:
        path: folder

    Returns:
        Sorted names of its subfolders
    """
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def count_png(root, leaf=None):
    """
    Count PNG files under a folder.

    Args:
        root: folder to walk
        leaf: if given, only count files in folders with this name (e.g. predicted_mask, accepted)

    Returns:
        Number of PNG files
    """
    n = 0
    for dirpath, _, files in os.walk(root):
        if leaf is None or os.path.basename(dirpath) == leaf:
            n += sum(f.lower().endswith(".png") for f in files)
    return n


def build_table(final_root, out_path, exclude=("Mouse-DSS",)):
    """
    Write the per-slide table: one row per slide from final/<protocol>/<slide>/final.txt,
    <protocol> <number of crypts> <8 colour columns>.

    Args:
        final_root: output root of step 4
        out_path: table file to write
        exclude: slide folder names to leave out

    Returns:
        List of the table rows
    """
    order = [p for p in PROTOCOLS if os.path.isdir(os.path.join(final_root, p))]
    order += [p for p in subfolders(final_root) if p not in PROTOCOLS]
    rows = []
    for protocol in order:
        for slide in subfolders(os.path.join(final_root, protocol)):
            path = os.path.join(final_root, protocol, slide, "final.txt")
            if slide in exclude or not os.path.isfile(path):
                continue
            # skip the header line; replace the slide name by the protocol
            for line in open(path).read().splitlines()[1:]:
                if line.strip():
                    rows.append(protocol + "\t" + "\t".join(line.split("\t")[1:]))
    with open(out_path, "w") as fh:
        fh.write("".join(r + "\n" for r in rows))
    return rows


def welch_pvalues(rows):
    """
    Welch t-test of Pressure-3min against each Boiling protocol, as in testing_data_analysis.ipynb: the
    per-slide value is brown top 25% minus brown bottom 25%, divided by the number of crypts.

    Args:
        rows: rows of the per-slide table

    Returns:
        List of (Boiling protocol, p-value)
    """
    from scipy.stats import ttest_ind
    values = {}
    for row in rows:
        f = row.split("\t")
        values.setdefault(f[0], []).append((float(f[8]) - float(f[9])) / float(f[1]))
    pressure = values.get("Pressure-3min", [])
    return [(b, ttest_ind(pressure, values[b], equal_var=False)[1]) for b in BOILING
            if len(values.get(b, [])) > 1 and len(pressure) > 1]


def check_inputs(args, src, dest, model, first, todo):
    """
    Stop with a message if an input of the steps that will run is missing.

    Args:
        args: parsed arguments
        src, dest, model: absolute paths of --src, --dest and --model
        first: index of the first step
        todo: steps that will run
    """
    if first > 0 and not os.path.isdir(os.path.join(dest, STAGE_DIR[STEPS[first - 1]])):
        sys.exit("--start %s needs %s" % (args.start, os.path.join(dest, STAGE_DIR[STEPS[first - 1]])))
    if "inference" in todo and not os.path.isfile(model):
        sys.exit("model not found: %s" % model)
    if ("inference" in todo or "ushape" in todo) and not os.path.isdir(src):
        sys.exit("tiles not found: %s" % src)


def clear_outputs(args, dest, todo):
    """
    Refuse to replace existing outputs of the steps that will run, unless --overwrite is given (then they
    are deleted).

    Args:
        args: parsed arguments
        dest: absolute output root
        todo: steps that will run
    """
    outputs = [os.path.join(dest, STAGE_DIR[s]) for s in todo if s in STAGE_DIR] + [os.path.join(dest, args.table)]
    existing = [p for p in outputs if (os.path.isdir(p) and os.listdir(p)) or os.path.isfile(p)]
    if existing and not args.overwrite:
        sys.exit("these outputs already exist (use another --dest, or --overwrite to replace them):\n  " + "\n  ".join(existing))
    for p in existing:
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def main():
    parser = argparse.ArgumentParser(
        description="TF inference pipeline: inference -> U-shape split -> U-shape extraction "
                    "-> colour detection -> per-slide table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", default="dataset/Whole_Slides_Segments_Processed/CDX2_Data",
                        help="colour-normalised, Hough-aligned tiles, <src>/<protocol>/<slide>/<tile>.png")
    parser.add_argument("--dest", required=True, help="output root, e.g. inference_run (gets inf/ us_split/ us_final/ final/)")
    parser.add_argument("--model", default="models/notransfer.h5", help="Keras weights")
    parser.add_argument("--table", default="res_new.txt", help="name of the per-slide table written in --dest")

    # Run control
    parser.add_argument("--start", default="inference", choices=STEPS,
                        help="first step to run; the folder of the step before it must already be in --dest")
    parser.add_argument("--jobs", type=int, default=4, help="parallel processes for steps 3 and 4 (one per protocol)")
    parser.add_argument("--overwrite", action="store_true", help="delete existing outputs of the steps that will run")
    args = parser.parse_args()

    # --- 1. Check the inputs and the outputs of the steps that will run ---
    src, dest, model = os.path.abspath(args.src), os.path.abspath(args.dest), os.path.abspath(args.model)
    first = STEPS.index(args.start)
    todo = STEPS[first:]
    check_inputs(args, src, dest, model, first, todo)
    clear_outputs(args, dest, todo)

    logs = os.path.join(dest, "logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    py = sys.executable
    t_start = time.time()

    # --- 2. Run the steps ---
    for step in todo:
        t0 = time.time()
        print("== %s" % step, flush=True)
        if step == "inference":
            # a relative --model is taken from the working directory of my_inference.py (the repository root)
            run_jobs([([py, os.path.join(HERE, "my_inference.py"), "--src", src + os.sep,
                        "--dest", os.path.join(dest, "inf") + os.sep, "--model", os.path.relpath(model, REPO)],
                       os.path.join(logs, "1_inference.log"))], 1, env)
            print("   %d crypt masks" % count_png(os.path.join(dest, "inf"), "predicted_mask"))
        elif step == "split":
            run_jobs([([py, os.path.join(HERE, "u_shape_split.py"), "--inf_root", os.path.join(dest, "inf"),
                        "--out_root", os.path.join(dest, "us_split")], os.path.join(logs, "2_split.log"))], 1, env)
            print("   " + ", ".join("%s %d" % (c, count_png(os.path.join(dest, "us_split"), c))
                                    for c in ("accepted", "fragmented", "rejected")))
        elif step == "ushape":
            # one process per protocol
            us_split = os.path.join(dest, "us_split")
            run_jobs([([py, os.path.join(HERE, "ushape_bottom_up.py"), "--us_split_root", os.path.join(us_split, p),
                        "--original_root", os.path.join(src, p), "--output_root", os.path.join(dest, "us_final", p)],
                       os.path.join(logs, "3_ushape_%s.log" % p)) for p in subfolders(us_split)], args.jobs, env)
            print("   %d crypt patches" % count_png(os.path.join(dest, "us_final")))
        elif step == "color":
            # one process per protocol
            us_final = os.path.join(dest, "us_final")
            run_jobs([([py, os.path.join(HERE, "color_detection.py"), "--src", os.path.join(us_final, p),
                        "--dest", os.path.join(dest, "final", p)], os.path.join(logs, "4_color_%s.log" % p))
                      for p in subfolders(us_final)], args.jobs, env)
        elif step == "table":
            rows = build_table(os.path.join(dest, "final"), os.path.join(dest, args.table))
            print("   %d slides -> %s" % (len(rows), os.path.join(dest, args.table)))
            for b, p in welch_pvalues(rows):
                print("   Pressure-3min vs %-14s Welch p = %.3g" % (b, p))
        print("   (%.1f min)" % ((time.time() - t0) / 60), flush=True)
    print("done in %.1f min; logs in %s" % ((time.time() - t_start) / 60, logs))


if __name__ == "__main__":
    main()


# Example usage (TF environment):
#
#   # whole pipeline on the processed tiles (about 40 min on CPU)
#   python src/inference/run_inference_pipeline.py --dest inference_run
#
#   # from the colour step on the stored results: reproduces inference/res_new.txt byte for byte
#   mkdir -p inference_check && ln -s ../inference/us_final inference_check/us_final
#   python src/inference/run_inference_pipeline.py --dest inference_check --start color
#
#   # only rebuild the table (and print the p-values) from inference/final
#   python src/inference/run_inference_pipeline.py --dest inference --start table --table res_new_rebuilt.txt
