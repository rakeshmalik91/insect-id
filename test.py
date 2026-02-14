"""
Incremental Classifier — Evaluation Script
==========================================

Paths are derived from --model-name (default: lepidoptera):
  dataset dir  : insect-dataset/{model_name}

Usage examples:

  1. Evaluate the latest checkpoint (auto-detected):
     python test.py

  2. Evaluate on multiple test directories with detailed output:
     python test.py --test-dirs insect-dataset/lepidoptera/test insect-dataset/src/test*/lepidoptera --print-no-match

  3. Evaluate with specific top-K values:
     python test.py --top-k 3 5 10 --print-preds

  4. Evaluate a specific epoch of the latest iteration:
     python test.py -e 10

  5. Evaluate a specific iteration and epoch:
     python test.py -i 1 -e 20

  6. Test a different model (e.g. odonata):
     python test.py -m odonata
"""

import os
import sys
import re
import argparse
import datetime
import glob


def _import_libs():
    """Lazy-import heavy dependencies so --help works without them."""
    global torch, test_top_k
    import torch as _torch
    torch = _torch
    from mynnlibv2 import test_top_k as _test_top_k
    test_top_k = _test_top_k





class TeeLogger:
    """Duplicates writes to both a stream (e.g. stdout) and a log file.
    ANSI escape codes are stripped from the file output."""

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        if '\r' not in message:
            self.log_file.write(self._ANSI_RE.sub("", message))
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


def setup_logging(log_suffix="lepidoptera", replace=False):
    """Tee stdout and stderr to logs/test.{log_suffix}.log (append mode or replace)."""
    log_file = f"logs/test.{log_suffix}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    mode = "w" if replace else "a"
    log_fh = open(log_file, mode, encoding="utf-8")
    
    sys.stdout = TeeLogger(sys.__stdout__, log_fh)
    sys.stderr = TeeLogger(sys.__stderr__, log_fh)
    
    print(f"\n{'#'*80}")
    print(f"# SESSION STARTED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# COMMAND: {' '.join(sys.argv)}")
    print(f"{'#'*80}\n")
    
    return log_fh


def find_latest_checkpoint(dataset_dir, model_name, version, iteration=None, epoch=None):
    """Find the latest checkpoint file.
    Filters by iteration and epoch if provided.
    Otherwise finds the latest (highest iter, highest epoch).
    """
    prefix = f"checkpoint.{model_name}.{version}"
    if not os.path.exists(dataset_dir):
        return None
    
    # Regex to extract iteration and epoch: ...i(\d+).e(\d+).pth
    pattern = re.compile(rf"checkpoint\.{re.escape(model_name)}\.{re.escape(version)}\.i(\d+)\.e(\d+)\.pth")
    
    best_ckpt = None
    best_iter = -1
    best_epoch = -1
    
    for fname in os.listdir(dataset_dir):
        match = pattern.match(fname)
        if match:
            it = int(match.group(1))
            ep = int(match.group(2))
            
            if iteration is not None and it != iteration:
                continue
            if epoch is not None and ep != epoch:
                continue
                
            # Logic to find "best" (highest iter, then highest epoch)
            # This works even if filtered (e.g. if filtered by iter, we find max epoch for that iter)
            
            is_better = False
            if it > best_iter:
                is_better = True
            elif it == best_iter:
                if ep > best_epoch:
                    is_better = True
            
            if is_better:
                best_iter = it
                best_epoch = ep
                best_ckpt = fname

    if best_ckpt:
        print(f"[INFO] Auto-detected checkpoint: {best_ckpt} (Iteration {best_iter}, Epoch {best_epoch})")
        return os.path.join(dataset_dir, best_ckpt)
        
    return None


def test(args, checkpoint_path=None):
    """Evaluate a checkpoint on one or more test directories."""
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(args.dataset_dir, args.model_name, args.version, args.iteration, args.epoch)
        
    if not checkpoint_path:
        msg = f"[ERROR] No checkpoint found in {args.dataset_dir} for model '{args.model_name}' (version {args.version})"
        if args.iteration:
            msg += f", iteration {args.iteration}"
        if args.epoch:
            msg += f", epoch {args.epoch}"
        print(msg)
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model_data = torch.load(checkpoint_path, weights_only=False)

    test_dirs = []
    if args.test_dirs:
        for t in args.test_dirs:
            matches = glob.glob(t)
            if matches:
                test_dirs.extend(matches)
            else:
                test_dirs.append(t)
    
    if not test_dirs:
        test_dirs = [f"{args.dataset_dir}/test"]

    aggregated_stats = {}

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"\n[WARNING] Test directory not found: {test_dir}. Skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"[INFO] Testing on: {test_dir}")
        print(f"{'='*60}")
        for k in args.top_k:
            stats = test_top_k(
                model_data,
                test_dir,
                k,
                print_preds=args.print_preds,

                # print_accuracy=True (default), so per-folder stats are printed
                print_top1_accuracy=(k == args.top_k[0]),
                print_no_match=args.print_no_match,
            )
            
            if k not in aggregated_stats:
                aggregated_stats[k] = {
                    'total_cnt': 0,
                    'top1_success_cnt': 0,
                    'success_cnt': 0,
                    'genus_success_cnt': 0,
                    'top1_genus_success_cnt': 0
                }
            
            s = aggregated_stats[k]
            s['total_cnt'] += stats['total_cnt']
            s['top1_success_cnt'] += stats['top1_success_cnt']
            s['success_cnt'] += stats['success_cnt']
            s['genus_success_cnt'] += stats['genus_success_cnt']
            s['top1_genus_success_cnt'] += stats['top1_genus_success_cnt']

    print(f"\n{'='*60}")
    print("[INFO] Combined Statistics")
    print(f"{'='*60}")
    for k in args.top_k:
        if k not in aggregated_stats:
            continue
        s = aggregated_stats[k]
        total = s['total_cnt']
        if total == 0:
            print(f"Top {k}: No samples.")
            continue
            
        print(f"Top   1 accuracy: {s['top1_success_cnt']}/{total} -> {100*s['top1_success_cnt']/total:.2f}%, genus matched: {s['top1_genus_success_cnt']}/{total} -> {100*s['top1_genus_success_cnt']/total:.2f}%")
        print(f"Top {k:3} accuracy: {s['success_cnt']}/{total} -> {100*s['success_cnt']/total:.2f}%, genus matched: {s['genus_success_cnt']}/{total} -> {100*s['genus_success_cnt']/total:.2f}%")
        print("-" * 10)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Incremental classifier — Evaluation"
    )
    parser.add_argument(
        "--model-name", "-m",
        default="lepidoptera",
        help="Model name — used to derive dataset dir (default: lepidoptera)",
    )
    parser.add_argument(
        "--version", "-v",
        default="v1",
        help="Version tag used for auto-discovery (default: v1)",
    )
    parser.add_argument(
        "--iteration", "-i",
        type=int,
        default=None,
        help="Specific iteration number to load (optional)",
    )
    parser.add_argument(
        "--epoch", "-e",
        type=int,
        default=None,
        help="Specific epoch number to load (optional)",
    )

    parser.add_argument(
        "--test-dirs",
        nargs="+",
        default=None,
        help="One or more test directories (default: <dataset-dir>/test)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[3],
        help="Top-K values to evaluate (default: 3)",
    )
    parser.add_argument("--print-preds", action="store_true", help="Print per-image predictions")
    parser.add_argument("--print-no-match", action="store_true", help="Print details for non-matching predictions")
    parser.add_argument("--log-replace", action="store_true", help="Replace log file instead of appending")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Derive paths from model name
    args.model_name = args.model_name.lower()
    args.dataset_dir = f"insect-dataset/{args.model_name}"
    
    # Find checkpoint early to determine log name
    ckpt_path = find_latest_checkpoint(args.dataset_dir, args.model_name, args.version, args.iteration, args.epoch)
    
    # Use model name and version for logging
    log_suffix = f"{args.model_name}.{args.version}"

    _import_libs()
    
    log_fh = setup_logging(log_suffix, replace=args.log_replace)

    try:
        test(args, checkpoint_path=ckpt_path)
    finally:
        # Restore sys.stdout/stderr before closing log file to avoid error during shutdown/exception printing
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()
