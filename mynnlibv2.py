import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from collections import defaultdict
import random
import os
from PIL import Image
import time
import datetime
import copy
try:
    from IPython.display import display
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
from pathlib import Path


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class IncrementalResNet(nn.Module):
    def __init__(self, num_classes):
        super(IncrementalResNet, self).__init__()
        base_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # remove fc
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)
        return logits

    def add_classes(self, num_new):
        old_weights = self.fc.weight.data.clone()
        old_bias = self.fc.bias.data.clone()
        new_fc = nn.Linear(2048, self.fc.out_features + num_new)
        new_fc.weight.data[:self.fc.out_features] = old_weights
        new_fc.bias.data[:self.fc.out_features] = old_bias
        self.fc = new_fc

def __get_transforms(phase, image_size, robustness):
    img_header_footer_ratio = 1.1
    normazile_x = [0.485, 0.456, 0.406]
    normalize_y = [0.229, 0.224, 0.225]
    if phase == 'train' and robustness < 0.5:
        return [
            transforms.Resize(int(image_size * img_header_footer_ratio)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomRotation(45 * robustness, fill=(0, 0, 0)),
            transforms.ColorJitter(brightness=0.2 * robustness, contrast=0.2 * robustness),
            transforms.ToTensor(),
            transforms.Normalize(normazile_x, normalize_y),
        ]
    elif phase == 'train':
        return [
            transforms.Resize(int(image_size * img_header_footer_ratio)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45 * robustness, fill=(0, 0, 0)),
            transforms.ColorJitter(brightness=0.2 * robustness, contrast=0.2 * robustness),
            transforms.ToTensor(),
            transforms.Normalize(normazile_x, normalize_y),
        ]
    else:
        return [
            transforms.Resize(int(image_size * img_header_footer_ratio)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(normazile_x, normalize_y),
        ]

def __validate_images(phase, phase_dir):
    num_images = 0
    for class_dir in os.listdir(phase_dir):
        class_path = f"{phase_dir}/{class_dir}"
        images = os.listdir(class_path)
        if not images:
            print(f"[WARNING] No images in: {class_path} — removing directory")
            try:
                os.rmdir(class_path)
            except OSError as e:
                print(f"[ERROR] Could not remove {class_path}: {e}")
            continue
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path)
                img.verify()
                if img.size[0] == 0 or img.size[1] == 0:
                    print("[WARNING] Zero size:", img_path, "— deleting")
                    os.remove(img_path)
                    continue
            except Exception as e:
                print("[WARNING] Corrupt:", img_path, e, "— deleting")
                try:
                    os.remove(img_path)
                except OSError:
                    pass
                continue
            
            num_images += 1
    print(f"[INFO] {phase} set: {num_images} images")

def __validate_images_for_all_phase(train_dir, val_dir):
    for phase in ['train', 'val']:
        phase_dir = train_dir if phase == 'train' else val_dir
        __validate_images(phase, phase_dir)
    
def validate_dataset(train_dir, val_dir):
    __validate_images_for_all_phase(train_dir, val_dir)

def __init_classes(model_data, train_dir):
    if 'class_names' not in model_data:
        model_data['class_names'] = os.listdir(train_dir)
        model_data['num_classes'] = len(model_data['class_names'])
        print(f"[INFO] classes: {model_data['num_classes']}")
    else:
        current_iteration_classes = os.listdir(train_dir)
        new_classes = [ cls for cls in current_iteration_classes if cls not in model_data['class_names'] ]
        model_data['class_names'] += new_classes
        model_data['num_classes'] = len(model_data['class_names'])
        model_data['num_new_classes'] = len(new_classes)
        print(f"[INFO] classes: total {model_data['num_classes']}, {len(current_iteration_classes)} in current iteration, {model_data['num_new_classes']} new")
    return model_data

def __init_model(model_data):
    model_data['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'model' not in model_data:
        model_data['model'] = IncrementalResNet(num_classes=model_data['num_classes'])
    else:
        model_data['model'].add_classes(num_new=model_data['num_new_classes'])

    model_data['model'].to(model_data['device'])

    return model_data

def __init_teacher_model(model_data):
    if model_data['iteration'] > 1:
        model_data['teacher_model'] = copy.deepcopy(model_data['model'])
        model_data['teacher_model'].eval()
        for p in model_data['teacher_model'].parameters():
            p.requires_grad = False
    return model_data

def init_model(train_dir, val_dir, batch_size=32, image_size=224, lr=1e-4, validate=True, num_workers=0):
    model_data = { 
        'version': 'v2', 
        'iteration': 1, 
        'epoch': 0,
        'batch_size': batch_size,
        'image_size': image_size,
        'image_size': image_size,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'num_workers': num_workers
    }

    model_data = __init_classes(model_data, train_dir)
    if validate:
        __validate_images_for_all_phase(train_dir, val_dir)
    model_data = __init_model(model_data)
    model_data['optimizer'] = torch.optim.Adam(model_data['model'].parameters(), lr=lr)
    model_data['criterion'] = nn.CrossEntropyLoss()

    return model_data

def init_iteration(model_data, train_dir, val_dir, lr=1e-4, validate=True, num_workers=None):
    model_data['iteration'] += 1
    model_data['epoch'] = 0
    del model_data['train_start_time']
    model_data['train_dir'], model_data['val_dir'] = train_dir, val_dir
    if num_workers is not None:
        model_data['num_workers'] = num_workers

    model_data = __init_classes(model_data, train_dir)
    if validate:
        __validate_images_for_all_phase(train_dir, val_dir)
    model_data = __init_model(model_data)
    model_data['optimizer'] = torch.optim.Adam(model_data['model'].parameters(), lr=lr)
    model_data['criterion'] = nn.CrossEntropyLoss()
    model_data = __init_teacher_model(model_data)

    if 'dataloaders' in model_data and 'train' in model_data['dataloaders'] and f"train_i{model_data['iteration'] - 1}" not in model_data['dataloaders']:
        model_data['dataloaders'][f"train_i{model_data['iteration'] - 1}"] = model_data['dataloaders']['train']

    if 'dataloaders' in model_data and 'val' in model_data['dataloaders'] and f"val_i{model_data['iteration'] - 1}" not in model_data['dataloaders']:
        model_data['dataloaders'][f"val_i{model_data['iteration'] - 1}"] = model_data['dataloaders']['val']

    return model_data

def __init_dataloaders(model_data, robustness=0.3):
    train_dir, val_dir = model_data['train_dir'], model_data['val_dir']
    batch_size, image_size = model_data['batch_size'], model_data['image_size']
    num_workers = model_data.get('num_workers', 0)

    if 'dataloaders' not in model_data:
        model_data['transform'], model_data['datasets'], model_data['dataloaders'] = {}, {}, {}

    train_class_cnt = len(os.listdir(train_dir))
    val_class_cnt = len(os.listdir(val_dir))
    if train_class_cnt != val_class_cnt:
        print(f"[WARNING] train class count ({val_class_cnt}) does not match val class count ({val_class_cnt})")

    for phase in ['train', 'val']:
        model_data['transform'][phase] = transforms.Compose(__get_transforms(phase, image_size, robustness))
        phase_dir = train_dir if phase == 'train' else val_dir
        image_data = [ 
            (f"{phase_dir}/{class_dir}/{img}", model_data['class_names'].index(class_dir))
            for class_dir in os.listdir(phase_dir) 
            for img in os.listdir(f"{phase_dir}/{class_dir}")
            if class_dir in model_data['class_names']
        ]
        model_data['datasets'][phase] = SimpleImageDataset(
            image_paths = [ img[0] for img in image_data],
            labels = [ img[1] for img in image_data],
            transform = model_data['transform'][phase]
        )
        model_data['dataloaders'][phase] = DataLoader(model_data['datasets'][phase], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return model_data

def __distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def __init_epoch_progress_bar(phase, model_data, dataloader):
    epoch_start_time = time.time()
    data_cnt = len(dataloader)
    data_idx = 0
    
    progress_data = {
        'epoch_start_time': epoch_start_time,
        'data_idx': data_idx,
        'data_cnt': data_cnt,
        'has_widgets': False,
        'phase': phase,
        'iteration': model_data.get('iteration', 0),
        'epoch': model_data.get('epoch', 0)
    }

    if not HAS_WIDGETS:
        return progress_data

    progress = widgets.IntProgress(
        value=0, min=0, max=data_cnt, 
        description=f"Iteration {model_data['iteration']:02} | Epoch {model_data['epoch']:02} | {phase.replace("_", " ").capitalize()} ", 
        layout=widgets.Layout(width='500px'),
        bar_style=('info' if phase=='train' else 'success')
    )
    progress.style = {'description_width': '250px'}
    label = widgets.Label(value=f"0/{data_cnt}")
    box = widgets.HBox([progress, label])
    display(box)
    return {
        'epoch_start_time': epoch_start_time,
        'data_idx': data_idx,
        'data_cnt': data_cnt,
        'progress': progress,
        'label': label,
        'box': box,
        'has_widgets': True
    }

def __update_epoch_progress_bar(progress_data, total_loss, total_correct, total_samples):
    progress_data['data_idx'] += 1
    
    # Pause check (Windows only)
    force_print = False
    if os.name == 'nt':
        import msvcrt
        if msvcrt.kbhit() and msvcrt.getch().lower() == b'p':
            print(f"\r[PAUSED] Press 'p' to resume...{' '*50}", end='', flush=True)
            pause_start = time.time()
            while True:
                if msvcrt.kbhit() and msvcrt.getch().lower() == b'p':
                    break
                time.sleep(0.1)
            pause_duration = time.time() - pause_start
            progress_data['epoch_start_time'] += pause_duration
            force_print = True
    
    if not progress_data.get('has_widgets', False):
        # Print only every 5 batches, on the last batch, or if forced (e.g. after pause)
        if force_print or progress_data['data_idx'] % 5 == 0 or progress_data['data_idx'] == progress_data['data_cnt']:
            elapsed = time.time() - progress_data['epoch_start_time']
            p_name = progress_data['phase'].replace('_', ' ').capitalize()
            msg = f"\r[INFO] Iteration {progress_data['iteration']:02} | Epoch {progress_data['epoch']:02} | {p_name:10} --> {progress_data['data_idx']}/{progress_data['data_cnt']} batches | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | Loss: {total_loss / total_samples:.3f} | Acc: {total_correct / total_samples:.3f}"
            print(msg, end='', flush=True)
        return

    elapsed = time.time() - progress_data['epoch_start_time']
    remaining = (elapsed / progress_data['data_idx']) * (progress_data['data_cnt'] - progress_data['data_idx'])
    progress_data['label'].value = f"{progress_data['progress'].value}/{progress_data['data_cnt']} batches | Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining))} | Loss: {total_loss / total_samples:.3} | Acc: {total_correct / total_samples:.3}"
    progress_data['progress'].value = progress_data['data_idx']

def __remove_epoch_progress_bar(phase, model_data, progress_data, total_loss, total_correct, total_samples):
    if progress_data.get('has_widgets', False):
        progress_data['box'].close()
    elapsed = time.time() - progress_data['epoch_start_time']
    print(f"\r", end="")
    print(f"[INFO] Iteration {model_data['iteration']:02} | Epoch {model_data['epoch']:02} | {phase.replace('_', ' ').capitalize():10} --> {progress_data['data_idx']}/{progress_data['data_cnt']} batches | Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | Loss: {total_loss / total_samples:.3f} | Acc: {total_correct / total_samples:.3f}")


def __extract_dataloader_subset(dataloader, dataset_subset_ratio):
    if dataset_subset_ratio >= 1.0:
        return dataloader
    dataset = dataloader.dataset
    total_size = len(dataset)
    subset_size = int(total_size * dataset_subset_ratio)
    subset_size = max(1, subset_size)
    indices = random.sample(range(total_size), subset_size)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)

def __merge_dataloaders(loader1, loader2):
    combined_dataset = ConcatDataset([loader1.dataset, loader2.dataset])
    return DataLoader(combined_dataset, batch_size=loader1.batch_size, shuffle=True)

def __run_epoch(phase, model_data, distill_lambda=1.0, temperature=2.0, replay_ratio=0.0):
    model_data['model'].train() if phase == 'train' else model_data['model'].eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    # Initialize scaler for mixed precision
    if 'scaler' not in model_data and phase == 'train':
        model_data['scaler'] = torch.amp.GradScaler('cuda')

    # merge replay data
    dataloader = model_data['dataloaders'][phase]
    if phase == 'train' and model_data['iteration'] > 1 and replay_ratio > 0.0:
        for i in range(model_data['iteration'] - 1, 0, -1):
            replay_key = f"train_i{i}"
            if replay_key in model_data['dataloaders']:
                replay_loader = __extract_dataloader_subset(model_data['dataloaders'][replay_key], replay_ratio)
                dataloader = __merge_dataloaders(dataloader, replay_loader)

    progress_data = __init_epoch_progress_bar(phase, model_data, dataloader)

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(model_data['device']), labels.to(model_data['device'])

        model_data['optimizer'].zero_grad()
        
        # Mixed Precision Training
        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'train':
                with torch.amp.autocast('cuda'):
                    outputs = model_data['model'](imgs)
                    loss = model_data['criterion'](outputs, labels)

                    # Add distillation loss if teacher is given
                    if 'teacher_model' in model_data:
                        with torch.no_grad():
                            teacher_outputs = model_data['teacher_model'](imgs)
                        loss += distill_lambda * __distillation_loss(outputs, teacher_outputs, temperature)
                
                # Backward pass with scaler
                model_data['scaler'].scale(loss).backward()
                model_data['scaler'].step(model_data['optimizer'])
                model_data['scaler'].update()
            else:
                # Validation (no scaler needed/used, but autocast helps speed)
                with torch.amp.autocast('cuda'):
                    outputs = model_data['model'](imgs)
                    loss = model_data['criterion'](outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

        __update_epoch_progress_bar(progress_data, total_loss, total_correct, total_samples)

    __remove_epoch_progress_bar(phase, model_data, progress_data, total_loss, total_correct, total_samples)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "model_data": model_data
    }

def run_epoch(model_data, output_path, robustness_lambda=0.05, replay_ratio=0):
    if 'train_start_time' not in model_data:
        model_data['train_start_time'] = time.time()
        model_data['elapsed_time'] = 0
        print(f"[INFO] Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    model_data['epoch'] += 1
    result = { 'model_data': model_data, 'epoch': model_data['epoch'], 'iteration': model_data['iteration'] }

    robustness = model_data['epoch'] * robustness_lambda
    model_data = __init_dataloaders(model_data, robustness=robustness)

    result['train_result'] = __run_epoch('train', model_data, replay_ratio=replay_ratio)
    model_data = result['train_result']['model_data']

    result['val_result'] = __run_epoch('val', model_data)

    if model_data['iteration'] > 1:
        for i in range(model_data['iteration'] - 1, 0, -1):
            if f"val_i{i}" in model_data['dataloaders']:
                result[f'val_i{i}_result'] = __run_epoch(f"val_i{i}", model_data)

    torch.save(model_data, f"{output_path}.i{model_data['iteration']:02}.e{model_data['epoch']:02}.pth")

    elapsed_time = time.time() - start_time
    model_data['elapsed_time'] += elapsed_time

    return result

def run_epochs(model_data, output_path, robustness_lambda=0.05, replay_ratio=0, replay_ratio_increment=0, max_replay_ratio=0, max_epochs=50, max_val_acc=0.95, max_val_acc_diff=0.001, stopping_threshold=3):
    results = []
    stopping_cnt = 0
    batch_start_time = time.time()

    for e in range(max_epochs):
        result = run_epoch(model_data, output_path, robustness_lambda=robustness_lambda, replay_ratio=clamp(replay_ratio + e * replay_ratio_increment, 0.0, max_replay_ratio))
        results.append(result)
        
        if results[e]['val_result']['acc'] >= max_val_acc:
            print(f"[INFO] Early stopping at {e+1} epochs with accuracy {result['val_result']['acc']:.3f}")
            break
        
        if e > 0 and results[e]['val_result']['acc'] - results[e-1]['val_result']['acc'] < max_val_acc_diff:
            stopping_cnt += 1
            if stopping_cnt >= stopping_threshold:
                print(f"[INFO] Stopping after {stopping_threshold} epochs with no significant improvement")
                break
            else:
                print(f"[INFO] ({stopping_cnt}/{stopping_threshold}) Early stopping imminent at {e+1} epochs with no significant improvement in accuracy ({results[e]['val_result']['acc']:.3f})")
        else:
            stopping_cnt = 0

    elapsed_time = time.time() - batch_start_time
    print(f"[INFO] Training completed after {len(results)} epochs in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    return results

def predict_top_k(image_path, model_data, k):
    model_data['model'].eval()
    image = Image.open(image_path).convert("RGB")
    image = model_data['transform']['val'](image).unsqueeze(0).to(model_data['device'])
    with torch.no_grad():
        outputs = model_data['model'](image)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k)
    try:
        return {model_data['class_names'][top_indices[0][i]]: top_probs[0][i].item() for i in range(0, k)}
    except Exception:
        return None

def test_top_k(model_data, test_dir, k, print_preds=True, print_accuracy=True, print_top1_accuracy=True, print_no_match=False, match_filter=0.0, print_genus_match=True):
    model_data['model'].eval()
    top1_success_cnt = 0
    top1_genus_success_cnt = 0
    success_cnt = 0
    genus_success_cnt = 0
    total_cnt = 0
    
    # Collect files and determine ground truth source
    test_path = Path(test_dir)
    items = [] # list of (file_path, ground_truth_label, is_valid_sample)
    
    # Check structure
    try:
        has_subdirs = any(p.is_dir() for p in test_path.iterdir())
    except Exception:
        has_subdirs = False
        
    if has_subdirs:
        # Directory structure: Class/Image.jpg
        for class_dir in test_path.iterdir():
            if class_dir.is_dir():
                ground_truth = class_dir.name
                for f in class_dir.iterdir():
                    if f.is_file() and not f.name.startswith('.'):
                        items.append((f, ground_truth, True))
    else:
        # Flat structure: Image_Species.jpg
        for f in test_path.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                is_valid = 'unidentified' not in f.name
                items.append((f, f.name, is_valid))

    max_name_len = max([len(f.name.split('.')[0]) for f, _, _ in items]) if items else 0

    for file, ground_truth, is_valid in items:
        if print_preds:
            print(f"{file.name.split('.')[0]:{max_name_len+1}}:", end=' ')
            
        if is_valid:
            total_cnt += 1
            
        probs = predict_top_k(file, model_data, k)
        if probs is None: continue # Skip if error opening image

        species_matched = False
        genus_matched = False
        
        # probs is ordered Top-1 to Top-K (assuming OrderedDict or Py3.7+ dict)
        top1_pred = list(probs.keys())[0] if probs else None
        
        for pred, prob in probs.items():
            if not match_filter or prob >= match_filter:
                # Check Match
                is_match = False
                if has_subdirs:
                     if pred == ground_truth: is_match = True
                else:
                     if pred in ground_truth: is_match = True
                     
                if is_match and is_valid:
                    species_matched = True
                    success_cnt += 1 # Count if ANY matches

                # Genus Logic
                if is_valid and pred.split('-')[0] in ground_truth:
                   genus_matched = True
                   
                if print_preds:
                    color = "\033[32m" if ((has_subdirs and pred == ground_truth) or (not has_subdirs and pred in ground_truth)) else ""
                    reset = "\033[0m"
                    print(f"{color}{pred}{reset}({prob:.3f}) ", end=' ')

        if print_preds:
             print() # Newline

        if not print_preds and print_no_match and not species_matched:
             print(f"{file.name.split('.')[0]:{max_name_len+1}}:", end=' ')
             for pred, prob in probs.items():
                 if not match_filter or prob >= match_filter:
                     # Color logic same as above
                     color = "\033[32m" if ((has_subdirs and pred == ground_truth) or (not has_subdirs and pred in ground_truth)) else "\033[31m" 
                     # Wait, if not species_matched, then NO match occurred (unless loose match logic differs?)
                     # species_matched is set if ANY top-k matches ground_truth.
                     # So if we are here, none matched.
                     # But maybe genus matched?
                     is_genus_match = pred.split('-')[0] in ground_truth
                     color = "\033[33m" if is_genus_match else "\033[31m"
                     
                     reset = "\033[0m"
                     print(f"{color}{pred}{reset}({prob:.3f}) ", end=' ')
             print()

        if genus_matched and is_valid:
             genus_success_cnt += 1
             
        # Top-1 Check
        if is_valid and top1_pred:
             is_top1_match = False
             if has_subdirs:
                  if top1_pred == ground_truth: is_top1_match = True
             else:
                  if top1_pred in ground_truth: is_top1_match = True
             
             if is_top1_match:
                  top1_success_cnt += 1
                  if top1_pred.split('-')[0] in ground_truth:
                       top1_genus_success_cnt += 1

    if print_accuracy and total_cnt > 0:
        if print_preds:
            print("-"*10)
        if print_top1_accuracy:
            p_str = f"Top   1 accuracy: {top1_success_cnt}/{total_cnt} -> {100*top1_success_cnt/total_cnt:.2f}%"
            if print_genus_match:
                 p_str += f", genus matched: {top1_genus_success_cnt}/{total_cnt} -> {100*top1_genus_success_cnt/total_cnt:.2f}%"
            print(p_str)
            
        p_str = f"Top {k:3} accuracy: {success_cnt}/{total_cnt} -> {100*success_cnt/total_cnt:.2f}%"
        if print_genus_match:
             p_str += f", genus matched: {genus_success_cnt}/{total_cnt} -> {100*genus_success_cnt/total_cnt:.2f}%"
        print(p_str)
             
    return {
        'total_cnt': total_cnt,
        'top1_success_cnt': top1_success_cnt,
        'success_cnt': success_cnt,
        'genus_success_cnt': genus_success_cnt,
        'top1_genus_success_cnt': top1_genus_success_cnt
    }


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))