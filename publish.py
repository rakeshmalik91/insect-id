"""
Publish Assets Script
=====================================================

This script automates the generation of assets for the Insect ID models, including:
1. Class details JSON (names, common names)
2. Image archives (for mobile app download)
3. TorchScript model conversion (for mobile app)
4. Metadata stats update (accuracy, counts)

Usage examples:

  1. Publish everything for a model (after training):
     python publish.py -m lepidoptera -v v2

  2. Only generate image archive:
     python publish.py -m odonata --task images

  3. Update stats only (validates model accuracy):
     python publish.py -m moth -v v1 --task stats

  4. Convert model to TorchScript:
     python publish.py -m cicada --task model

  5. Convert model to ExecuTorch format (.pte):
     python publish.py -m lepidoptera -v v2 --task model --executorch
"""

import os
import sys
import shutil
import json
import argparse
import random
import re
import time
import zipfile
import torch
from PIL import Image
from itertools import chain
from datetime import datetime
from pathlib import Path

# Add project root to sys.path if needed
sys.path.append(os.getcwd())

# Try to import shared configurations and functions from train.py
try:
    from train import MODEL_TYPE_MAP, SOURCE_MAP, IGNORED_SOURCES, find_latest_checkpoint
except ImportError:
    print("[WARNING] Could not import from train.py. Using local definitions.")
    MODEL_TYPE_MAP = {
        "lepidoptera": ["moth", "butterfly"],
        "odonata": ["odonata"],
        "cicada": ["cicada"],
        "non_lepidoptera": ["odonata", "cicada", "misc"],
    }
    SOURCE_MAP = {
        "moth": ['inaturalist.org', 'mothsofindia.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "butterfly": ['ifoundbutterflies.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "odonata": ['indianodonata.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "cicada": ['indiancicadas.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "misc": ['inaturalist.org'],
        "root-classifier": []
    }
    IGNORED_SOURCES = ["insecta.pro", "wikipedia.org", "indianbiodiversity.org"]
    
    def find_latest_checkpoint(dataset_dir, model_name, version):
        prefix = f"checkpoint.{model_name}.{version}"
        if not os.path.exists(dataset_dir):
            return None
        checkpoints = [f for f in os.listdir(dataset_dir) if f.startswith(prefix) and f.endswith(".pth")]
        if not checkpoints:
            return None
        checkpoints.sort()
        return os.path.join(dataset_dir, checkpoints[-1])

import mynnlibv2

# --- Logging Setup ---

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
    """Tee stdout and stderr to logs/publish.{model_name}.{version}.log."""
    log_file = f"logs/publish.{model_name}.{version}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fh = open(log_file, "a", encoding="utf-8")
    
    sys.stdout = TeeLogger(sys.__stdout__, log_fh)
    sys.stderr = TeeLogger(sys.__stderr__, log_fh)
    
    print(f"\n{'#'*80}")
    print(f"# SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# COMMAND: {' '.join(sys.argv)}")
    print(f"{'#'*80}\n")
    
    return log_fh

# --- Constants ---
MAX_IMG_CNT = 6
MAX_IMG_SIZE = 300
OUTPUT_DIR = "models"
METADATA_PATH = "insect-id-app/metadata.json"

# --- Helper Functions for Image Processing ---

def is_black(pixel):
    threshold = 10
    return pixel[0] <= threshold and pixel[1] <= threshold and pixel[2] <= threshold

def crop_header_footer(img):
    width, height = img.size
    start = int(height * 0.15)
    # Scan from top
    try:
        while start > 0 and not is_black(img.getpixel((int(width / 2), start))):
            start -= 1
    except IndexError: pass
        
    end = int(height * 0.85)
    # Scan from bottom
    try:
        while end < height - 1 and not is_black(img.getpixel((int(width / 2), end))):
             end += 1
    except IndexError: pass
        
    return img.crop((0, start, width, end))

def center_crop(img):
    width, height = img.size
    square_size = min(width, height)
    left = (width - square_size) / 2
    top = (height - square_size) / 2
    right = (width + square_size) / 2
    bottom = (height + square_size) / 2
    return img.crop((left, top, right, bottom))

def resize_img(img):
    width, height = img.size
    img_size = min(width, MAX_IMG_SIZE)
    return img.resize((img_size, img_size), Image.LANCZOS)

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def zip_folder(folder_path, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    shutil.rmtree(folder_path)

# --- Asset Generation Functions ---

def generate_class_details(model_name, output_model_name=None):
    if not output_model_name: output_model_name = model_name
    print(f"[INFO] Generating class details for {output_model_name} (from {model_name})...")
    
    # Define input mapping based on model type
    input_files = {}
    if model_name == "lepidoptera":
        input_files = {
            "insect-dataset/src/class_details.ifoundbutterflies.org.json": "Butterfly",
            "insect-dataset/src/class_details.mothsofindia.org.json": "Moth",
        }
    elif model_name == "odonata":
        input_files = {
            "insect-dataset/src/class_details.indianodonata.org.json": "Odonata"
        }
    elif model_name == "cicada":
        input_files = {
            "insect-dataset/src/class_details.indiancicadas.org.json": "Cicada"
        }
    elif model_name == "moth":
         input_files = {
            "insect-dataset/src/class_details.mothsofindia.org.json": "Moth"
        }
    elif model_name == "butterfly":
         input_files = {
            "insect-dataset/src/class_details.ifoundbutterflies.org.json": "Butterfly"
        }
    
    combined_data = {}
    
    for src, suffix in input_files.items():
        if not os.path.exists(src):
            print(f"[WARNING] Source file not found: {src}")
            continue
            
        data = load_json(src)
        for key, value in data.items():
            # Clean up name: append suffix if not present
            if 'name' in value and not re.match(r"(?i)^.*(moth|moths|butterfly|butterflies|fly|flies|cicada|cicadas|dragonfly|damselfly)$", value['name']):
                value['name'] += " " + suffix
            
            # Remove image URLs (as per notebook)
            if 'images' in value:
                del value['images']
                
            combined_data[key] = value

    output_path = f"{OUTPUT_DIR}/class_details.{output_model_name}.json"
    dump_json(output_path, combined_data)
    print(f"[INFO] Saved class details to {output_path}")


def generate_image_archive(model_name, output_image_archive_name=None, overwrite=False):
    if not output_image_archive_name: output_image_archive_name = model_name
    print(f"[INFO] Generating image archive for {output_image_archive_name} (from {model_name})...")
    
    types = MODEL_TYPE_MAP.get(model_name, [model_name])
    
    # Collect source directories (checking for duplicates)
    src_dirs = []
    seen_dirs = set()
    
    def add_dir(path):
        # Normalize path to ensure uniqueness checks are robust
        norm_path = os.path.normpath(path)
        if norm_path not in seen_dirs and os.path.exists(path):
            src_dirs.append(path)
            seen_dirs.add(norm_path)

    # 1. Add Source directories for all types
    for t in types:
        sources = SOURCE_MAP.get(t, [])
        for src in sources:
            p1 = f"insect-dataset/src/{src}"
            p2 = f"insect-dataset/src/{t}.{src}" # Try prefixed
            
            add_dir(p1)
            add_dir(p2)
            
    # 2. Add the model's own data directory
    add_dir(f"insect-dataset/{model_name}/data")

    # 3. Add directories for each component type data
    for t in types:
        add_dir(f"insect-dataset/{t}/data")

    print(f"[INFO] Consolidating images from: {src_dirs}")
    
    dst_root = f"{OUTPUT_DIR}/images.{output_image_archive_name}"
    
    if overwrite:
        if os.path.exists(f"{dst_root}.zip"):
            print(f"[INFO] Removing existing archive {dst_root}.zip due to --overwrite.")
            os.remove(f"{dst_root}.zip")
        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
            
    if os.path.exists(f"{dst_root}.zip"):
        print(f"[INFO] Archive {dst_root}.zip already exists. Unzipping to update...")
        with zipfile.ZipFile(f"{dst_root}.zip", 'r') as z:
            z.extractall(dst_root)
            
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Process images
    # Collect all classes
    all_classes = set()
    for d in src_dirs:
        if os.path.exists(d):
            all_classes.update(os.listdir(d))
            
    species_added = 0
    
    for class_name in all_classes:
        class_dst = f"{dst_root}/{class_name}"
        if os.path.exists(class_dst):
            continue # Skip if already has images
            
        os.makedirs(class_dst, exist_ok=True)
        species_added += 1
        
        count = 0
        # Iterate sources to find images
        for src_dir in src_dirs:
            src_class_path = f"{src_dir}/{class_name}"
            if not os.path.exists(src_class_path):
                continue
                
            for file_name in os.listdir(src_class_path):
                file_path = f"{src_class_path}/{file_name}"
                if not os.path.isfile(file_path): continue
                
                try:
                    img = Image.open(file_path).convert("RGB")
                    img = crop_header_footer(img)
                    img = center_crop(img)
                    img = resize_img(img)
                    
                    if img.size[0] >= MAX_IMG_SIZE: # Only save if large enough? Notebook check: if img.size[0] < max_img_size: continue
                         img.save(f"{class_dst}/{count + 1}.jpg", "JPEG", quality=50)
                         count += 1
                except Exception as e:
                    # print(f"Error processing {file_path}: {e}")
                    pass
                
                if count >= MAX_IMG_CNT: break
            if count >= MAX_IMG_CNT: break
            
        if count == 0:
            os.rmdir(class_dst) # Remove empty dir
            species_added -= 1
            
    print(f"[INFO] Added {species_added} new species to archive.")
    
    # Zip result
    if os.path.exists(dst_root):
        zip_folder(dst_root, f"{dst_root}.zip")
        print(f"[INFO] Created archive: {dst_root}.zip")


def convert_and_update_model(model_name, checkpoint_path=None, version="v1", use_executorch=False, output_model_name=None, overwrite=False):
    if not output_model_name:
        output_model_name = model_name
        
    print(f"[INFO] Converting model {model_name} to {output_model_name}...")
    dataset_dir = f"insect-dataset/{model_name}"
    
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(dataset_dir, model_name, version)
        
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found for {model_name} (version {version}). Skipping.")
        return

    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    
    # Check if files exist to avoid overwriting
    output_classes = f"{OUTPUT_DIR}/classes.{output_model_name}.json"
    
    if use_executorch:
        output_model = f"{OUTPUT_DIR}/m.checkpoint.{output_model_name}.pte"
    else:
        output_model = f"{OUTPUT_DIR}/m.checkpoint.{output_model_name}.pt"
        
    if os.path.exists(output_model) and not overwrite:
        print(f"[WARNING] Model file {output_model} already exists. Use --overwrite to replace. Skipping conversion.")
        return
        
    if os.path.exists(output_classes) and not overwrite:
        print(f"[WARNING] Classes file {output_classes} already exists. Use --overwrite to replace. Skipping conversion.")
        return

    # Load and Prepare Model
    model_data = torch.load(checkpoint_path, weights_only=False)
    model = model_data['model']
    model.eval()
    model.to("cpu")
    
    # Save classes
    dump_json(output_classes, model_data['class_names'])
    print(f"[INFO] Saved classes to {output_classes}")

    if model_name == "root-classifier":
        print(f"[INFO] Updating root-classifier classes in metadata...")
        metadata = load_json(METADATA_PATH)
        if 'root-classifier' not in metadata: metadata['root-classifier'] = {}
        metadata['root-classifier']['classes'] = model_data['class_names']
        dump_json(METADATA_PATH, metadata)
        print(f"[INFO] Updated {METADATA_PATH} with root-classifier classes")

    if use_executorch:
        try:
            import executorch.exir as exir
            from torch.export import export
            from executorch.exir import to_edge
            
            print("[INFO] converting to ExecuTorch (pte)...")
            example_args = (torch.randn(1, 3, 224, 224),)
            
            # 1. torch.export (pre-autograd ATen dialect)
            captured_model = export(model, example_args)
            
            # 2. to_edge (Edge dialect)
            edge_program = to_edge(captured_model)
            
            # 3. to_executorch (ExecuTorch backend)
            executorch_program = edge_program.to_executorch()
            
            with open(output_model, "wb") as f:
                f.write(executorch_program.buffer)
            print(f"[INFO] Saved ExecuTorch model to {output_model}")
            
        except ImportError:
            print("[ERROR] ExecuTorch not installed. Please install 'executorch'.")
        except Exception as e:
            print(f"[ERROR] Failed to convert to ExecuTorch: {e}")
            # cleanup classes if model failed?
            
    else:
        # Default: TorchScript
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_model)
            print(f"[INFO] Saved TorchScript model to {output_model}")
            
        except Exception as e:
            print(f"[ERROR] Failed to convert model: {e}")
            return


def update_stats(model_name, version="v1", output_model_name=None, output_image_archive_name=None):
    if not output_model_name: output_model_name = model_name
    if not output_image_archive_name: output_image_archive_name = model_name
    
    print(f"[INFO] Updating stats for {output_model_name} (archive: {output_image_archive_name})...")
    dataset_dir = f"insect-dataset/{model_name}"
    data_dir = f"{dataset_dir}/data"
    val_dir = f"{dataset_dir}/val"
    
    metadata = load_json(METADATA_PATH)
    if output_model_name not in metadata:
        metadata[output_model_name] = {}
    
    # 1. Dataset stats
    if os.path.exists(data_dir):
        if 'stats' not in metadata[output_model_name] or not metadata[output_model_name]['stats']:
            metadata[output_model_name]['stats'] = {}
            
        stats = metadata[output_model_name]['stats']
        all_classes = os.listdir(data_dir)
        stats['class_count'] = len(all_classes)
        stats['species_count'] = len([c for c in all_classes if not re.match(r"^.*-(early|genera|spp)$", c)])
        stats['spp_class_count'] = len([c for c in all_classes if re.match(r"^.*-(genera|spp)$", c)])
        stats['early_stage_class_count'] = len([c for c in all_classes if re.match(r"^.*-(early)$", c)])
        stats['data_count'] = sum([len(os.listdir(f"{data_dir}/{c}")) for c in all_classes])
        
    # 2. Model Size
    metadata[output_model_name]['size'] = 0
    files_to_check = [
        f"{OUTPUT_DIR}/m.checkpoint.{output_model_name}.pt",
        f"{OUTPUT_DIR}/images.{output_image_archive_name}.zip",
        f"{OUTPUT_DIR}/classes.{output_model_name}.json",
        f"{OUTPUT_DIR}/class_details.{output_model_name}.json"
    ]
    for file in files_to_check:
        if os.path.exists(file):
            metadata[output_model_name]['size'] += os.path.getsize(file)
            
    # 3. Accuracy (Top 1 / Top 3)
    checkpoint_path = find_latest_checkpoint(dataset_dir, model_name, version)
    if checkpoint_path and os.path.exists(val_dir):
        print(f"[INFO] Validating accuracy using {checkpoint_path} on {val_dir}...")
        model_data = torch.load(checkpoint_path, weights_only=False)
        
        # We can use mynnlibv2.test_top_k helper or similar
        # But mynnlibv2.test_top_k prints outputs. We want return values.
        # It returns a dict with counts.
        
        # Ensure model is on device (cpu or cuda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_data['model'].to(device)
        model_data['device'] = device
        
        # Run K=3 to get both Top-1 and Top-3 stats in one pass
        # test_top_k returns 'top1_success_cnt' (always Top-1) and 'success_cnt' (Top-K)
        res = mynnlibv2.test_top_k(model_data, val_dir, k=3, print_preds=False, print_accuracy=False)
        
        top1_acc = res['top1_success_cnt'] / res['total_cnt'] if res['total_cnt'] > 0 else 0
        top3_acc = res['success_cnt'] / res['total_cnt'] if res['total_cnt'] > 0 else 0
        
        print(f"Top 1 Accuracy: {top1_acc*100:.2f}%")
        print(f"Top 3 Accuracy: {top3_acc*100:.2f}%")
        
        stats = metadata[output_model_name].get('stats', {})
        stats['accuracy'] = f"{top1_acc*100:.2f}%"
        stats['accuracy_top3'] = f"{top3_acc*100:.2f}%"
        metadata[output_model_name]['stats'] = stats

    dump_json(METADATA_PATH, metadata)
    print(f"[INFO] Updated metadata at {METADATA_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Publish assets for Insect ID models")
    parser.add_argument("--model-name", "-m", required=True, help="Model name (e.g. lepidoptera, moth, odonata)")
    parser.add_argument("--version", "-v", default="v1", help="Model version for checkpoint finding")
    parser.add_argument("--checkpoint-path", "-c", help="Specific checkpoint path (optional)")
    parser.add_argument("--task", "-t", default="all", choices=["all", "class-details", "images", "model", "stats"], help="Task to perform")
    parser.add_argument("--executorch", action="store_true", help="Convert to ExecuTorch (.pte) instead of TorchScript (.pt)")
    parser.add_argument("--output-model-name", help="Custom output name for model and classes file (default: model-name)")
    parser.add_argument("--output-image-archive-name", help="Custom output name for image archive (default: model-name)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    log_fh = setup_logging(args.model_name, args.version)
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if args.task in ["all", "class-details"]:
            generate_class_details(args.model_name, output_model_name=args.output_model_name)
            
        if args.task in ["all", "images"]:
            generate_image_archive(args.model_name, output_image_archive_name=args.output_image_archive_name, overwrite=args.overwrite)
            
        if args.task in ["all", "model"]:
            convert_and_update_model(args.model_name, args.checkpoint_path, args.version, use_executorch=args.executorch, output_model_name=args.output_model_name, overwrite=args.overwrite)
            
        if args.task in ["all", "stats"]:
            update_stats(args.model_name, args.version, output_model_name=args.output_model_name, output_image_archive_name=args.output_image_archive_name)
            
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()

if __name__ == "__main__":
    main()
