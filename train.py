"""
Incremental Classifier — Training Script
=====================================================

Paths are derived from --model-name (default: lepidoptera) and --version (default: v1):
  dataset dir  : insect-dataset/{model_name}
  checkpoint   : insect-dataset/{model_name}/checkpoint.{model_name}.{version}

The script automatically aggregates data from 'insect-dataset/src/*' into 'insect-dataset/{model_name}/data'
before every training run. If new images are detected while resuming, an incremental learning iteration is started automatically.

Usage examples:

  1. Train a model (starts from scratch; auto-aggregates data):
     python train.py

  2. Train with custom hyperparameters:
     python train.py --lr 5e-5 --batch-size 64 --max-epochs 20 --robustness-lambda 0.05

  3. Resume training (auto-detected):
     python train.py --max-epochs 5

  4. Train with automatic early stopping:
     python train.py --auto-stop --max-val-acc 0.96 --max-epochs 50

  5. Train a different model (e.g. odonata):
    python train.py -m odonata

  6. Train with cool-down (e.g. 5 minutes between epochs):
     python train.py -m lepidoptera -v v2 --max-epochs 5 --skip-aggregate --skip-validate --cool-down 5
"""

import os
import sys
import re
import argparse
import shutil
import random
import datetime
import json
import time


def _import_libs():
    """Lazy-import heavy dependencies so --help works without them."""
    global torch, validate_dataset, init_model, init_iteration, run_epoch, run_epochs
    import torch as _torch
    torch = _torch
    from mynnlibv2 import (
        validate_dataset as _validate_dataset,
        init_model as _init_model,
        init_iteration as _init_iteration,
        run_epoch as _run_epoch,
        run_epochs as _run_epochs,
    )
    validate_dataset = _validate_dataset
    init_model = _init_model
    init_iteration = _init_iteration
    run_epoch = _run_epoch
    run_epochs = _run_epochs







MODEL_TYPE_MAP = {
    "lepidoptera": ["moth", "butterfly"],
    "odonata": ["odonata"],
    "cicada": ["cicada"],
}

SOURCE_MAP = {
    "moth": ['inaturalist.org', 'mothsofindia.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
    "butterfly": ['ifoundbutterflies.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
    "odonata": ['indianodonata.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
    "cicada": ['indiancicadas.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org']
}

IGNORED_SOURCES = ["insecta.pro", "wikipedia.org", "indianbiodiversity.org"]



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


def setup_logging(model_name="lepidoptera", version="v1"):
    """Tee stdout and stderr to logs/train.{model_name}.{version}.log (append mode)."""
    log_file = f"logs/train.{model_name}.{version}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fh = open(log_file, "a", encoding="utf-8")
    
    sys.stdout = TeeLogger(sys.__stdout__, log_fh)
    sys.stderr = TeeLogger(sys.__stderr__, log_fh)
    
    print(f"\n{'#'*80}")
    print(f"# SESSION STARTED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# COMMAND: {' '.join(sys.argv)}")
    print(f"{'#'*80}\n")
    
    return log_fh


def normalize_data(dataset_dir, min_val_data_cnt=2):
    """Ensure validation set has at least `min_val_data_cnt` images per class
    by copying from the training data directory, and remove val classes that
    don't exist in the training data."""

    train_data_dir = f"{dataset_dir}/data"
    val_dir = f"{dataset_dir}/val"

    # Add missing val images
    for class_name in os.listdir(train_data_dir):
        class_val_dir = f"{val_dir}/{class_name}"
        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)
        val_data_cnt = len(os.listdir(class_val_dir))
        data_to_add = max(0, min_val_data_cnt - val_data_cnt)
        if data_to_add > 0:
            files = os.listdir(f"{train_data_dir}/{class_name}")
            random.shuffle(files)
            for file in files[:data_to_add]:
                shutil.copy2(
                    f"{train_data_dir}/{class_name}/{file}",
                    f"{class_val_dir}/{file}",
                )

    # Remove val classes not present in training data
    for class_name in os.listdir(val_dir):
        if not os.path.exists(f"{train_data_dir}/{class_name}"):
            shutil.rmtree(f"{val_dir}/{class_name}")
            print(f"[WARNING] class {class_name} in val not present in data — removed")



def load_valid_species(model_name):
    """Load valid species set from species.json."""
    if not os.path.exists("species.json"):
        print("[WARNING] species.json not found. Skipping species validation.")
        return None
        
    try:
        with open("species.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if model_name not in data:
            print(f"[WARNING] Model '{model_name}' not found in species.json.")
            return None
            
        valid_set = set()
        for group in data[model_name].get("species", []):
            for name in group.get("names", []):
                valid_set.add(name)
        
        if valid_set:
            print(f"[INFO] Loaded {len(valid_set)} valid species from species.json")
        return valid_set
    except Exception as e:
        print(f"[ERROR] Failed to load species.json: {e}")
        return None


def aggregate_data(model_name, dataset_dir):
    """Copy images from source directories to the model's data directory.
    Returns True if any new data was added."""
    data_dir = f"{dataset_dir}/data"
    os.makedirs(data_dir, exist_ok=True)
    
    types = MODEL_TYPE_MAP.get(model_name.lower(), [model_name.lower()])
    added_files = 0
    
    print(f"[INFO] Aggregating data for model '{model_name}' (types: {types})...")
    
    valid_species = load_valid_species(model_name)

    for insect_type in types:
        sources = SOURCE_MAP.get(insect_type, [])
        for source in sources:
            if source in IGNORED_SOURCES:
                continue
                
            source_path = f"insect-dataset/src/{source}"
            if not os.path.exists(source_path):
                continue
            
            print(f"  - Processing source: {source} ({insect_type})")
            
            # Walk through species directories in the source
            for species_name in os.listdir(source_path):
                src_species_dir = f"{source_path}/{species_name}"
                if not os.path.isdir(src_species_dir):
                    continue
                
                # Filter by valid species (allow suffixes)
                if valid_species is not None:
                    is_valid =  species_name in valid_species
                    if not is_valid and '-' in species_name:
                        parts = species_name.split("-")
                        for i in range(len(parts)-1, 0, -1):
                            prefix = "-".join(parts[:i])
                            if prefix in valid_species:
                                is_valid = True
                                break
                    if not is_valid:
                        continue
                        
                dst_species_dir = f"{data_dir}/{species_name}"
                os.makedirs(dst_species_dir, exist_ok=True)
                
                # Copy files
                for file_name in os.listdir(src_species_dir):
                    src_file = f"{src_species_dir}/{file_name}"
                    dst_file = f"{dst_species_dir}/{file_name}"
                    
                    if not os.path.isfile(src_file):
                        continue

                    if not os.path.exists(dst_file):
                        try:
                            shutil.copy2(src_file, dst_file)
                            added_files += 1
                        except Exception as e:
                            print(f"[WARNING] Failed to copy {src_file}: {e}")
                        
    if added_files > 0:
        print(f"[INFO] Added {added_files} new images to {data_dir}")
        return True
    
    print("[INFO] No new images found in sources.")
    return False


def find_latest_checkpoint(dataset_dir, model_name, version):
    """Find the latest checkpoint file (highest iter/epoch) for the model."""
    prefix = f"checkpoint.{model_name}.{version}"
    if not os.path.exists(dataset_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(dataset_dir) 
        if f.startswith(prefix) and f.endswith(".pth")
    ]
    if not checkpoints:
        return None
        
    # Sort lexicographically (works because iXX and eXX are zero-padded)
    checkpoints.sort()
    latest = checkpoints[-1]
    print(f"[INFO] Auto-detected latest checkpoint: {latest}")
    return os.path.join(dataset_dir, latest)


def train(args):
    """Run training (initial or incremental iteration)."""
    dataset_dir = args.dataset_dir
    checkpoint_path = args.checkpoint_path
    
    # 1. Aggregate data from sources
    has_new_data = False
    if not args.skip_aggregate:
        has_new_data = aggregate_data(args.model_name, dataset_dir)
    else:
        print("[INFO] Skipping data aggregation...")
    
    # 2. Normalize and validate dataset
    normalize_data(dataset_dir)
    
    if not args.skip_validate:
        validate_dataset(f"{dataset_dir}/data", f"{dataset_dir}/val")
    else:
        print("[INFO] Skipping dataset validation...")

    # 3. Initialize model / load checkpoint
    resume_path = find_latest_checkpoint(dataset_dir, args.model_name, args.version)
    
    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model_data = torch.load(resume_path, weights_only=False)
        
        # If new data was added, start a new iteration automatically
        if has_new_data:
            print(f"[INFO] New data detected! Starting new incremental iteration (i{model_data['iteration']+1})...")
            # Create iteration-specific directories if needed, or just use current data/val
            # The notebook used separate dirs (i01-train), here we use the main data/val 
            # as the 'current' dataset for this new iteration.
            model_data = init_iteration(
                model_data, 
                f"{dataset_dir}/data", 
                f"{dataset_dir}/val", 
                lr=args.lr,
                num_workers=args.num_workers
            )
        else:
            print(f"[INFO] No new data. Resuming current iteration (i{model_data['iteration']})...")
            
    else:
        print("[INFO] No checkpoint found. Starting fresh training (Iteration 1)...")
        model_data = init_model(
            f"{dataset_dir}/data",
            f"{dataset_dir}/val",
            batch_size=args.batch_size,
            image_size=args.image_size,
            lr=args.lr,
            validate=False,
            num_workers=args.num_workers,
        )

    # 4. Run training loop
    model_data['num_workers'] = args.num_workers
    model_data['batch_size'] = args.batch_size
    if args.auto_stop:
        run_epochs(
            model_data,
            output_path=checkpoint_path,
            robustness_lambda=args.robustness_lambda,
            replay_ratio=args.replay_ratio,
            max_epochs=args.max_epochs,
            max_val_acc=args.max_val_acc,
            max_val_acc_diff=args.max_val_acc_diff,
            stopping_threshold=args.stopping_threshold,
        )
    else:
        for _ in range(args.max_epochs):
            run_epoch(
                model_data,
                output_path=checkpoint_path,
                robustness_lambda=args.robustness_lambda,
                replay_ratio=args.replay_ratio,
            )
            if args.cool_down > 0:
                sys.__stdout__.write(f"[INFO] Cooling down for {args.cool_down} minutes...\n")
                time.sleep(args.cool_down * 60)





def build_parser():
    parser = argparse.ArgumentParser(
        description="Incremental classifier — Training"
    )
    parser.add_argument(
        "--model-name", "-m",
        default="lepidoptera",
        help="Model name — used to derive dataset dir and checkpoint path (default: lepidoptera)",
    )
    parser.add_argument(
        "--version", "-v",
        default="v1",
        help="Version tag for checkpoint naming (default: v1)",
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--image-size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum number of epochs (default: 15)")
    parser.add_argument(
        "--robustness-lambda",
        type=float,
        default=0.1,
        help="Robustness lambda — augmentation increases by this factor each epoch (default: 0.1)",
    )
    parser.add_argument(
        "--replay-ratio",
        default=0.0,
        help="Replay ratio for experience replay (default: 0.0)",
    )
    parser.add_argument("--cool-down", type=float, default=0.0, help="Wait time in minutes between epochs (default: 0)")

    # Auto-stop parameters (used with --auto-stop flag)
    parser.add_argument("--auto-stop", action="store_true", help="Enable automatic early stopping via run_epochs")
    parser.add_argument("--max-val-acc", type=float, default=0.95, help="Target val accuracy for early stopping (default: 0.95)")
    parser.add_argument("--max-val-acc-diff", type=float, default=0.001, help="Min val accuracy improvement (default: 0.001)")
    parser.add_argument("--stopping-threshold", type=int, default=3, help="Epochs without improvement before stopping (default: 3)")
    
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip data aggregation from sources")
    parser.add_argument("--skip-validate", action="store_true", help="Skip dataset validation (corrupt check)")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Derive paths from model name
    args.model_name = args.model_name.lower()
    args.dataset_dir = f"insect-dataset/{args.model_name}"
    args.checkpoint_path = f"insect-dataset/{args.model_name}/checkpoint.{args.model_name}.{args.version}"

    _import_libs()
    log_fh = setup_logging(args.model_name, args.version)

    try:
        train(args)
    finally:
        # Restore sys.stdout/stderr before closing log file
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()
