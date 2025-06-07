import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from collections import defaultdict
import random
import os
from PIL import Image
import time
import datetime
import copy


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
        class_path = os.path.join(phase_dir, class_dir)
        images = os.listdir(class_path)
        if not images:
            print(f"[WARNING] No images in: {class_path}")
            continue
        for img_name in images:
            num_images += 1
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path)
                img.verify()
                if img.size[0] == 0 or img.size[1] == 0:
                    print("[WARNING] Zero size:", img_path)
            except Exception as e:
                print("[WARNING] Corrupt:", img_path, e)
    print(f"[INFO] {phase} set: {num_images} images")

def __validate_images_for_all_phase(train_dir, val_dir):
    for phase in ['train', 'val']:
        phase_dir = train_dir if phase == 'train' else val_dir
        __validate_images(phase, phase_dir)

def __init_classes(model_data, train_dir):
    if 'class_names' not in model_data:
        model_data['class_names'] = os.listdir(train_dir)
        model_data['num_classes'] = len(model_data['class_names'])
        print(f"[INFO] classes: {model_data['num_classes']}")
    else:
        new_classes = [ cls for cls in os.listdir(train_dir) if cls not in model_data['class_names'] ]
        model_data['class_names'] += new_classes
        model_data['num_classes'] = len(model_data['class_names'])
        model_data['num_new_classes'] = len(new_classes)
        print(f"[INFO] classes: {model_data['num_classes']}, added {model_data['num_new_classes']} new classes")
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

def init_model(train_dir, val_dir, batch_size=32, image_size=224, lr=1e-4):
    model_data = { 
        'version': 'v2', 
        'iteration': 1, 
        'epoch': 0,
        'batch_size': batch_size,
        'image_size': image_size,
        'train_dir': train_dir,
        'val_dir': val_dir
    }

    model_data = __init_classes(model_data, train_dir)
    __validate_images_for_all_phase(train_dir, val_dir)
    model_data = __init_model(model_data)
    model_data['optimizer'] = torch.optim.Adam(model_data['model'].parameters(), lr=lr)
    model_data['criterion'] = nn.CrossEntropyLoss()

    return model_data

def init_iteration(model_data, train_dir, val_dir, lr=1e-4):
    model_data['iteration'] += 1
    model_data['epoch'] = 0
    del model_data['train_start_time']
    model_data['train_dir'], model_data['val_dir'] = train_dir, val_dir

    model_data = __init_classes(model_data, train_dir)
    __validate_images_for_all_phase(train_dir, val_dir)
    model_data = __init_model(model_data)
    model_data['optimizer'] = torch.optim.Adam(model_data['model'].parameters(), lr=lr)
    model_data['criterion'] = nn.CrossEntropyLoss()
    model_data = __init_teacher_model(model_data)

    if 'dataloaders' in model_data and 'val' in model_data['dataloaders']:
        model_data['dataloaders']['old_val'] = model_data['dataloaders']['val']

    return model_data

def __init_dataloaders(model_data, robustness=0.3):
    train_dir, val_dir = model_data['train_dir'], model_data['val_dir']
    batch_size, image_size = model_data['batch_size'], model_data['image_size']

    if 'dataloaders' not in model_data:
        model_data['transform'], model_data['datasets'], model_data['dataloaders'] = {}, {}, {}

    for phase in ['train', 'val']:
        model_data['transform'][phase] = transforms.Compose(__get_transforms(phase, image_size, robustness))
        phase_dir = train_dir if phase == 'train' else val_dir
        image_data = [ 
            (f"{phase_dir}/{class_dir}/{img}", model_data['class_names'].index(class_dir))
            for class_dir in os.listdir(phase_dir) 
            for img in os.listdir(f"{phase_dir}/{class_dir}")
        ]
        model_data['datasets'][phase] = SimpleImageDataset(
            image_paths = [ img[0] for img in image_data],
            labels = [ img[1] for img in image_data],
            transform = model_data['transform'][phase]
        )
        model_data['dataloaders'][phase] = DataLoader(model_data['datasets'][phase], batch_size=batch_size, shuffle=True)

    return model_data

def __distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def __run_epoch(phase, model_data, distill_lambda=1.0, temperature=2.0):
    model_data['model'].train() if phase == 'train' else model_data['model'].eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for imgs, labels in model_data['dataloaders'][phase]:
        imgs, labels = imgs.to(model_data['device']), labels.to(model_data['device'])

        model_data['optimizer'].zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model_data['model'](imgs)
            loss = model_data['criterion'](outputs, labels)

            # Add distillation loss if teacher is given
            if 'teacher_model' in model_data and phase == 'train':
                with torch.no_grad():
                    teacher_outputs = model_data['teacher_model'](imgs)
                loss += distill_lambda * __distillation_loss(outputs, teacher_outputs, temperature)

            if phase == 'train':
                loss.backward()
                model_data['optimizer'].step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "model_data": model_data
    }

def run_epoch(model_data, output_path, robustness_lambda=0.05):
    if 'train_start_time' not in model_data:
        model_data['train_start_time'] = time.time()
        model_data['elapsed_time'] = 0
        print(f"[INFO] Training started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    start_time = time.time()
    model_data['epoch'] += 1
    print(f"[INFO] I{model_data['iteration']:02}.E{model_data['epoch']:02} | ", end='')

    robustness = model_data['epoch'] * robustness_lambda
    model_data = __init_dataloaders(model_data, robustness=robustness)

    train_result = __run_epoch('train', model_data)
    print(f"Train Loss={train_result['loss']:.3f} Acc={train_result['acc']:.3f} | ", end='')
    model_data = train_result['model_data']

    val_result = __run_epoch('val', model_data)
    print(f"Val Loss={val_result['loss']:.3f} Acc={val_result['acc']:.3f} | ", end='')

    if model_data['iteration'] > 1 and 'old_val' in model_data['dataloaders']:
        old_val_result = __run_epoch('old_val', model_data)
        print(f"Old Val Loss={old_val_result['loss']:.3f} Acc={old_val_result['acc']:.3f} | ", end='')

    torch.save(model_data, f"{output_path}.i{model_data['iteration']:02}.e{model_data['epoch']:02}.pth")

    elapsed_time = time.time() - start_time
    model_data['elapsed_time'] += elapsed_time
    print(f"Elapsed {datetime.timedelta(seconds=model_data['elapsed_time'])}")

    return model_data
