"""
Colon Crypt Segmentation (PyTorch) - Whole Inference Pipeline in One Command

This script runs the whole PyTorch inference pipeline (TF parity) on colour-normalised, Hough-aligned
tiles (dataset/Whole_Slides_Segments_Processed/CDX2_Data). It calls the step scripts of
pytorch_mrcnn/src/ in order and writes the same folders as the TF pipeline (and as
pytorch_mrcnn/inference/):

    1 inference  src/inference/my_inference.py       -> <dest>/inf/<protocol>/<slide>/{predicted_images,predicted_mask}/
    2 split      src/inference/u_shape_split.py      -> <dest>/us_split/<protocol>/<slide>/{accepted,fragmented,rejected}/
    3 ushape     src/inference/ushape_bottom_up.py   -> <dest>/us_final/<protocol>/<slide>/{accepted,fragmented}/
    4 color      src/inference/color_detection.py    -> <dest>/final/<protocol>/<slide>/final.txt (+ mask_and/, pdfs)
    5 table      src/analysis/build_res_table.py     -> <dest>/res_tfparity.txt (one row per slide, Mouse-DSS left out)
    6 plot       src/analysis/plot_pvalues.py        -> <dest>/plots/human_pytorch_tfparity.png and the Welch p-values

If the TF results are present, src/validation/compare_tf_pytorch_results.py then compares the run with
them (-> <dest>/comparison_with_tf.json). Steps 3 and 4 run one process per protocol; the output of
every step goes to <dest>/logs/. --start <step> reuses the folder of the step before it from <dest>.
Existing results are never replaced unless --overwrite is given.

Usage (torch_env, GPU node, from pytorch_mrcnn/: relative paths, including the defaults of --src and --weights,
are taken from the working directory):
    python src/inference/run_inference_pipeline.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \\
        --dest inference_run --weights models/notransfer_torch.pth

Device (--device): cuda for an NVIDIA GPU (default; the stored results were made on one), cpu for a computer
without an NVIDIA GPU, e.g. a Mac, and mps for the GPU of an Apple-silicon Mac (not tested). Without an NVIDIA
GPU, add --device cpu (or --device mps) to the command above.

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
PT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # pytorch_mrcnn/
REPO = os.path.abspath(os.path.join(PT_ROOT, ".."))         # repository root (TF results in inference/)
STEPS = ["inference", "split", "ushape", "color", "table", "plot"]
STAGE_DIR = {"inference": "inf", "split": "us_split", "ushape": "us_final", "color": "final"}
PLOT_NAME = "human_pytorch_tfparity.png"
COMPARE_LINES = ("crypt masks", "slides whose", "U-shape patches", "per-slide value", "Welch", "   Boiling")


def run_jobs(jobs, n_parallel, env):
    """
    Run step commands from pytorch_mrcnn/, at most n_parallel at a time; stop everything if one fails.

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
            running.append((subprocess.Popen(cmd, cwd=PT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
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
        Sorted names of its subfolders (the protocols)
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


def tail(path, n):
    """
    Args:
        path: text file
        n: number of lines

    Returns:
        The last n lines, without line ends
    """
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh.readlines()[-n:]]


def check_inputs(args, src, dest, weights, todo):
    """
    Stop with a message if an input of the steps that will run is missing.

    Args:
        args: parsed arguments
        src, dest, weights: absolute paths of --src, --dest and --weights
        todo: steps that will run
    """
    need = {"split": "inf", "ushape": "us_split", "color": "us_final", "table": "final", "plot": args.table}.get(args.start)
    if need and not os.path.exists(os.path.join(dest, need)):
        sys.exit("--start %s needs %s" % (args.start, os.path.join(dest, need)))
    if "inference" in todo and not os.path.isfile(weights):
        sys.exit("weights not found: %s" % weights)
    if ("inference" in todo or "ushape" in todo) and not os.path.isdir(src):
        sys.exit("tiles not found: %s" % src)


def clear_outputs(args, dest, todo, compare):
    """
    Refuse to replace existing outputs of the steps that will run, unless --overwrite is given (then they
    are deleted).

    Args:
        args: parsed arguments
        dest: absolute output root
        todo: steps that will run
        compare: whether the comparison with the TF results will run
    """
    outputs = [os.path.join(dest, STAGE_DIR[s]) for s in todo if s in STAGE_DIR]
    if "table" in todo:
        outputs.append(os.path.join(dest, args.table))
    if "plot" in todo:
        outputs.append(os.path.join(dest, "plots", PLOT_NAME))
    if compare:
        outputs.append(os.path.join(dest, "comparison_with_tf.json"))
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
        description="PyTorch pipeline with TF parity: inference -> U-shape split -> U-shape "
                    "extraction -> colour detection -> per-slide table -> p-values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", default="dataset/Whole_Slides_Segments_Processed/CDX2_Data",
                        help="colour-normalised, Hough-aligned tiles, <src>/<protocol>/<slide>/<tile>.png")
    parser.add_argument("--dest", required=True, help="output root, e.g. inference_run (gets inf/ us_split/ us_final/ final/)")
    parser.add_argument("--weights", default="models/notransfer_torch.pth", help="converted state dict")
    parser.add_argument("--table", default="res_tfparity.txt", help="name of the per-slide table written in --dest")

    # Inference options
    parser.add_argument("--device", default="cuda",
                        help="cuda: NVIDIA GPU (default); cpu: no NVIDIA GPU, e.g. a Mac; "
                             "mps: Apple-silicon GPU on a Mac (not tested)")
    parser.add_argument("--skip", nargs="*", default=["Mouse-DSS"], help="slide folders left out of the inference")
    parser.add_argument("--stages_dir", default=None, help="optional: save per-tile stage outputs of step 1 here (validation)")

    # Comparison with the TF results
    parser.add_argument("--tf_results", default=os.path.join(REPO, "inference"),
                        help="TF results to compare with at the end (default: <repository>/inference)")
    parser.add_argument("--no_compare", action="store_true", help="skip the comparison with the TF results")

    # Run control
    parser.add_argument("--start", default="inference", choices=STEPS,
                        help="first step to run; the folder of the step before it must already be in --dest")
    parser.add_argument("--jobs", type=int, default=4, help="parallel processes for steps 3 and 4 (one per protocol)")
    parser.add_argument("--overwrite", action="store_true", help="delete existing outputs of the steps that will run")
    args = parser.parse_args()

    # --- 1. Check the inputs and the outputs of the steps that will run ---
    src, dest, weights = os.path.abspath(args.src), os.path.abspath(args.dest), os.path.abspath(args.weights)
    first = STEPS.index(args.start)
    todo = STEPS[first:]
    check_inputs(args, src, dest, weights, todo)
    compare = not args.no_compare and os.path.isdir(os.path.join(args.tf_results, "inf"))
    clear_outputs(args, dest, todo, compare)

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
            cmd = [py, os.path.join(HERE, "my_inference.py"), "--src", src, "--dest", os.path.join(dest, "inf"),
                   "--weights", weights, "--device", args.device, "--skip"] + args.skip
            if args.stages_dir:
                cmd += ["--stages_dir", os.path.abspath(args.stages_dir)]
            run_jobs([(cmd, os.path.join(logs, "1_inference.log"))], 1, env)
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
            print("   %d crypt patches (every accepted/fragmented crypt has one)" % count_png(os.path.join(dest, "us_final")))
        elif step == "color":
            # one process per protocol
            us_final = os.path.join(dest, "us_final")
            run_jobs([([py, os.path.join(HERE, "color_detection.py"), "--src", os.path.join(us_final, p),
                        "--dest", os.path.join(dest, "final", p)], os.path.join(logs, "4_color_%s.log" % p))
                      for p in subfolders(us_final)], args.jobs, env)
        elif step == "table":
            log = os.path.join(logs, "5_table.log")
            run_jobs([([py, os.path.join(PT_ROOT, "src", "analysis", "build_res_table.py"), "--final_root",
                        os.path.join(dest, "final"), "--out", os.path.join(dest, args.table)], log)], 1, env)
            print("   " + tail(log, 1)[0])
        elif step == "plot":
            log = os.path.join(logs, "6_plot.log")
            run_jobs([([py, os.path.join(PT_ROOT, "src", "analysis", "plot_pvalues.py"), "--res",
                        os.path.join(dest, args.table), "--out", os.path.join(dest, "plots", PLOT_NAME)], log)], 1, env)
            for line in tail(log, 4):
                print("   " + line.strip())
        print("   (%.1f min)" % ((time.time() - t0) / 60), flush=True)

    # --- 3. Compare with the TF results ---
    if compare:
        print("== compare with the TF results in %s" % args.tf_results, flush=True)
        log = os.path.join(logs, "7_compare_with_tf.log")
        run_jobs([([py, os.path.join(PT_ROOT, "src", "validation", "compare_tf_pytorch_results.py"), "--a", args.tf_results,
                    "--a_name", "TF", "--b", dest, "--b_name", "PyTorch", "--json",
                    os.path.join(dest, "comparison_with_tf.json")], log)], 1, env)
        for line in open(log).read().splitlines():
            if line.startswith(COMPARE_LINES):
                print("   " + line.strip())
    print("done in %.1f min; logs in %s" % ((time.time() - t_start) / 60, logs))


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/, torch_env, GPU node):
#
#   # whole pipeline on the processed tiles, compared with the TF results in ../inference at the end
#   python src/inference/run_inference_pipeline.py --dest inference_run
#
#   # the same on a computer without an NVIDIA GPU, e.g. a Mac
#   python src/inference/run_inference_pipeline.py --dest inference_run --device cpu
#
#   # exact reproduction check: also save the stage outputs, then compare with the stored results
#   python src/inference/run_inference_pipeline.py --dest inference_run --stages_dir inference_run_stages
#
#   # re-run from the colour step on an existing run (keeps inf/, us_split/, us_final/)
#   python src/inference/run_inference_pipeline.py --dest inference_run --start color --overwrite
