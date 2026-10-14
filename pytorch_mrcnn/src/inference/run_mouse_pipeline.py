"""
Colon Crypt Segmentation (PyTorch) - Mouse DSS Pipeline in One Command

This script runs the PyTorch pipeline on the patches of the mouse DSS experiment (boiling against pressure) and
writes the same folders and tables as the TF results in ../inference/Mouse. It starts from the stored
colour-normalised patches <mouse_root>/<set>/patch_X_nm (X = b for boiling, p for pressure), which it links into
--dest instead of copying them, and calls the step scripts of pytorch_mrcnn/src/:

    0 inputs     <dest>/<set>/patch_X_nm -> relative link to <mouse_root>/<set>/patch_X_nm
    1 inference  src/inference/my_inference.py            -> <dest>/<set>/patch_X_inf/{predicted_images,predicted_mask}/
    2 split      src/inference/u_shape_split.py           -> <dest>/<set>/patch_X_split/{accepted,fragmented,rejected}/
    3 ushape     src/inference/ushape_bottom_up_mouse.py  -> <dest>/<set>/patch_X_us/{accepted,fragmented}/
                 (the U-shape step with exact tile matching; the mouse patch names make the original step cut some
                 crypts from the wrong patch)
    4 color      src/inference/color_detection.py         -> <dest>/<set>/final/final.txt (+ mask_and/, PDFs)
    5 tables     <dest>/res_mouse.txt (the two final.txt rows, named boiling and pressure) and
                 src/analysis/mouse_patch_statistics.py   -> per_image_brown_intensity.tsv, summary_and_pvalue.tsv,
                 notebook_pvalue.tsv, violin plots (with the patch selection of the stored TF analysis, by file name)

The output of every step goes to <dest>/logs/. --start <step> reuses the folders of the steps before it from
<dest>. Existing outputs are never replaced unless --overwrite is given. Steps 2-4 run one process per set.

Usage (torch_env, GPU node, from pytorch_mrcnn/):
    python src/inference/run_mouse_pipeline.py --dest inference/Mouse

Author: Amitash Nanda
Date: 2026-10-13
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
REPO = os.path.abspath(os.path.join(PT_ROOT, ".."))         # repository root (TF mouse results in inference/Mouse)
SETS = [("boiling", "b"), ("pressure", "p")]
STEPS = ["inference", "split", "ushape", "color", "tables"]
STEP_DIR = {"inference": "inf", "split": "split", "ushape": "us"}     # patch_X_<dir>
TABLES = ["res_mouse.txt", "per_image_brown_intensity.tsv", "per_image_brown_intensity_all.tsv", "summary_and_pvalue.tsv",
          "notebook_pvalue.tsv", "violin_boiling_vs_pressure.png", "violin_boiling_vs_pressure_notebook.png"]


def run_jobs(jobs, env):
    """
    Run step commands from pytorch_mrcnn/ in parallel; stop everything if one fails.

    Args:
        jobs: list of (command, log file) pairs
        env: environment of the processes
    """
    running = []
    for cmd, log_path in jobs:
        log = open(log_path, "w")
        running.append((subprocess.Popen(cmd, cwd=PT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
    failed = None
    for proc, log, log_path in running:
        proc.wait()
        log.close()
        if proc.returncode != 0 and failed is None:
            failed = (proc.returncode, log_path)
    if failed:
        sys.exit("step failed (exit code %d), see %s" % failed)


def folder(dest, set_name, x, step):
    """
    Args:
        dest: output root
        set_name, x: 'boiling', 'b' or 'pressure', 'p'
        step: 'nm', 'inf', 'split', 'us' or 'final'

    Returns:
        <dest>/<set>/patch_X_<step> (or <dest>/<set>/final)
    """
    return os.path.join(dest, set_name, "final" if step == "final" else "patch_%s_%s" % (x, step))


def link_inputs(mouse_root, dest):
    """
    Link the stored colour-normalised patches into --dest (relative links, nothing is copied).

    Args:
        mouse_root: TF mouse results with <set>/patch_X_nm/
        dest: output root
    """
    for set_name, x in SETS:
        src = folder(mouse_root, set_name, x, "nm")
        link = folder(dest, set_name, x, "nm")
        if not os.path.isdir(src):
            sys.exit("input patches not found: %s" % src)
        if os.path.lexists(link):
            if os.path.realpath(link) != os.path.realpath(src):
                print("   note: %s exists and is used as the input (it is not a link to %s)" % (link, src))
            continue
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.relpath(src, os.path.dirname(link)), link)
        print("   %s -> %s" % (link, os.readlink(link)))


def clear_outputs(dest, todo, overwrite):
    """
    Refuse to replace existing outputs of the steps that will run, unless overwrite (then they are deleted).

    Args:
        dest: output root
        todo: steps that will run
        overwrite: delete existing outputs
    """
    outputs = []
    for step in todo:
        if step in STEP_DIR:
            outputs += [folder(dest, s, x, STEP_DIR[step]) for s, x in SETS]
        elif step == "color":
            outputs += [folder(dest, s, x, "final") for s, x in SETS]
        elif step == "tables":
            outputs += [os.path.join(dest, t) for t in TABLES]
    existing = [p for p in outputs if os.path.lexists(p)]
    if existing and not overwrite:
        sys.exit("these outputs already exist (use another --dest, or --overwrite to replace them):\n  " + "\n  ".join(existing))
    for p in existing:
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def write_res_mouse(dest):
    """
    Write <dest>/res_mouse.txt: the second line of each final.txt, with the folder name replaced by the set name.

    Args:
        dest: output root

    Returns:
        The two rows
    """
    rows = []
    for set_name, x in SETS:
        with open(os.path.join(folder(dest, set_name, x, "final"), "final.txt")) as fh:
            row = fh.read().splitlines()[1].split("\t")
        rows.append("\t".join([set_name] + row[1:]))
    with open(os.path.join(dest, "res_mouse.txt"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch pipeline on the mouse DSS patches: inference -> U-shape split -> U-shape extraction "
                    "(exact tile matching) -> colour detection -> res_mouse.txt and per-patch statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--mouse_root", default=os.path.join(REPO, "inference", "Mouse"),
                        help="TF mouse results with <set>/patch_X_nm/ (default: <repository>/inference/Mouse)")
    parser.add_argument("--dest", required=True, help="output root, e.g. inference/Mouse (gets boiling/ pressure/ and the tables)")
    parser.add_argument("--weights", default=os.path.join(PT_ROOT, "models", "notransfer_torch.pth"),
                        help="converted state dict (default: pytorch_mrcnn/models/notransfer_torch.pth)")
    parser.add_argument("--device", default="cuda",
                        help="cuda: NVIDIA GPU (default); cpu: no NVIDIA GPU, e.g. a Mac; "
                             "mps: Apple-silicon GPU on a Mac (not tested)")
    parser.add_argument("--selection", default=os.path.join(PT_ROOT, "src", "analysis", "mouse_patch_selection.tsv"),
                        help="patch selection list of the statistics step")

    # Run control
    parser.add_argument("--start", default="inference", choices=STEPS,
                        help="first step to run; the folders of the steps before it must already be in --dest")
    parser.add_argument("--no_plot", action="store_true", help="no violin plots")
    parser.add_argument("--overwrite", action="store_true", help="delete existing outputs of the steps that will run")
    args = parser.parse_args()

    # --- 1. Inputs, and the outputs of the steps that will run ---
    dest, mouse_root, weights = os.path.abspath(args.dest), os.path.abspath(args.mouse_root), os.path.abspath(args.weights)
    todo = STEPS[STEPS.index(args.start):]
    if "inference" in todo and not os.path.isfile(weights):
        sys.exit("weights not found: %s" % weights)
    need = {"split": "inf", "ushape": "split", "color": "us", "tables": "final"}.get(args.start)
    if need and not all(os.path.isdir(folder(dest, s, x, need)) for s, x in SETS):
        sys.exit("--start %s needs %s" % (args.start, folder(dest, "<set>", "X", need)))
    print("== inputs (colour-normalised patches of %s)" % mouse_root)
    link_inputs(mouse_root, dest)
    clear_outputs(dest, todo, args.overwrite)
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
            # one set after the other on the GPU
            for s, x in SETS:
                run_jobs([([py, os.path.join(HERE, "my_inference.py"), "--src", folder(dest, s, x, "nm"),
                            "--dest", folder(dest, s, x, "inf"), "--weights", weights, "--device", args.device],
                           os.path.join(logs, "1_inference_%s.log" % s))], env)
        elif step == "split":
            run_jobs([([py, os.path.join(HERE, "u_shape_split.py"), "--inf_root", folder(dest, s, x, "inf"),
                        "--out_root", folder(dest, s, x, "split")], os.path.join(logs, "2_split_%s.log" % s))
                      for s, x in SETS], env)
        elif step == "ushape":
            run_jobs([([py, os.path.join(HERE, "ushape_bottom_up_mouse.py"), "--us_split_root", folder(dest, s, x, "split"),
                        "--original_root", folder(dest, s, x, "nm"), "--output_root", folder(dest, s, x, "us")],
                       os.path.join(logs, "3_ushape_%s.log" % s)) for s, x in SETS], env)
        elif step == "color":
            run_jobs([([py, os.path.join(HERE, "color_detection.py"), "--src", folder(dest, s, x, "us"),
                        "--dest", folder(dest, s, x, "final")], os.path.join(logs, "4_color_%s.log" % s))
                      for s, x in SETS], env)
        elif step == "tables":
            for row in write_res_mouse(dest):
                print("   res_mouse.txt: %s ..." % "\t".join(row.split("\t")[:2]))
            log = os.path.join(logs, "5_tables.log")
            cmd = [py, os.path.join(PT_ROOT, "src", "analysis", "mouse_patch_statistics.py"),
                   "--boiling_us", folder(dest, "boiling", "b", "us"), "--pressure_us", folder(dest, "pressure", "p", "us"),
                   "--out", dest, "--root", dest, "--selection", os.path.abspath(args.selection)]
            run_jobs([(cmd + (["--no_plot"] if args.no_plot else []), log)], env)
            for line in open(log).read().splitlines():
                if line.startswith("[INFO]"):
                    print("   " + line[7:])
        if step in ("inference", "split", "ushape"):
            for s, x in SETS:
                d = folder(dest, s, x, STEP_DIR[step])
                n = sum(len([f for f in fn if f.endswith(".png")]) for _, _, fn in os.walk(d))
                print("   %-8s %4d PNG files in %s" % (s, n, os.path.relpath(d, dest)))
        print("   (%.1f min)" % ((time.time() - t0) / 60), flush=True)
    print("done in %.1f min; logs in %s" % ((time.time() - t_start) / 60, logs))


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/, torch_env, GPU node):
#
#   # the stored PyTorch mouse results (pytorch_mrcnn/inference/Mouse, from ../inference/Mouse/<set>/patch_X_nm)
#   python src/inference/run_mouse_pipeline.py --dest inference/Mouse
#
#   # a new run into another folder, then only the tables again
#   python src/inference/run_mouse_pipeline.py --dest mouse_run
#   python src/inference/run_mouse_pipeline.py --dest mouse_run --start tables --overwrite
