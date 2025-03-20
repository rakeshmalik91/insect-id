import shutil
import os
import time
import datetime
import re
import random
import numpy as np
from pathlib import Path
from PIL import Image
import pprint
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import json

def split_data_for_train_and_val(data_dir, test_dir, val_dir, train_dir, test_data_weight=0.1, val_data_weight=0.2, min_file_cnt_for_val=4, class_name_filter_regex=None):
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    train_data_cnt = 0
    val_data_cnt = 0
    test_data_cnt = 0
    class_cnt = 0
    
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir() and os.listdir(class_dir):
            class_dir_name = class_dir.name
            if class_name_filter_regex and not re.match(class_name_filter_regex, class_dir_name, re.IGNORECASE):
                continue
            class_cnt = class_cnt + 1
            file_count = sum(1 for file in class_dir.iterdir() if file.is_file())
            for file in Path(class_dir).iterdir():
                if file.is_file():
                    random_float = random.random()
                    if file_count >= min_file_cnt_for_val and random_float < test_data_weight:
                        target_dir = test_dir
                        test_data_cnt = test_data_cnt + 1
                    elif file_count >= min_file_cnt_for_val and random_float < test_data_weight + val_data_weight:
                        target_dir = val_dir
                        val_data_cnt = val_data_cnt + 1
                    else:
                        target_dir = train_dir
                        train_data_cnt = train_data_cnt + 1
                    target_dir_path = f"{target_dir}/{class_dir_name}"
                    if not os.path.exists(target_dir_path):
                        os.makedirs(target_dir_path)
                    shutil.copy(file, target_dir_path)

    print(f"Class count: {class_cnt}")
    print(f"Total data count: {train_data_cnt+val_data_cnt+test_data_cnt}")
    print(f"Training data count: {train_data_cnt}")
    print(f"Validation data count: {val_data_cnt}")
    print(f"Test data count: {test_data_cnt}")

def match_val_class_to_idx_with_train(model_data, silent=False):
    if not silent:
        print(f"train class count: {len(model_data['datasets']['train'].class_to_idx)}")
        print(f"val class count: {len(model_data['datasets']['val'].class_to_idx)}")
    if len(model_data['datasets']['val'].class_to_idx) != len(model_data['datasets']['train'].class_to_idx):
        model_data['datasets']['val'].class_to_idx =  model_data['datasets']['train'].class_to_idx
        new_val_samples = []
        for path, label in model_data['datasets']['val'].samples:
            class_name = model_data['datasets']['val'].classes[label]
            if class_name in model_data['datasets']['train'].class_to_idx:
                new_val_samples.append((path, model_data['datasets']['train'].class_to_idx[class_name]))
        model_data['datasets']['val'].samples = new_val_samples
    return model_data
    
def get_train_transforms(image_size, robustness):
    if robustness < 0.5:
        return [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomRotation(45*robustness),
            transforms.ColorJitter(brightness=0.2*robustness, contrast=0.2*robustness),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        return [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45*robustness),
            transforms.ColorJitter(brightness=0.2*robustness, contrast=0.2*robustness),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]


def prepare_dataloaders(model_data, train_dir, val_dir, batch_size, image_size, robustness, silent=True):
    model_data['transform'] = {
        'train': transforms.Compose(get_train_transforms(image_size, robustness)),
        'val': transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    model_data['datasets'] = {
        'train': datasets.ImageFolder(root=train_dir, transform=model_data['transform']['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=model_data['transform']['val']),
    }
    model_data = match_val_class_to_idx_with_train(model_data, silent)
    model_data['dataloaders'] = {
        'train': DataLoader(model_data['datasets']['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(model_data['datasets']['val'], batch_size=batch_size, shuffle=False),
    }
    return model_data

def init_model_for_training(train_dir, val_dir, batch_size=32, arch='resnet18', image_size=224, robustness=0.3, lr=0.001, weight_decay=None, label_smoothing=None, silent=False):
    model_data = {}
    model_data = prepare_dataloaders(model_data, train_dir, val_dir, batch_size, image_size, robustness, silent)
    model_data['class_names'] = model_data['datasets']['train'].classes

    if arch == 'resnet152':
        model_data['model'] = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif arch == 'resnet101':
        model_data['model'] = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif arch == 'resnet50':
        model_data['model'] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif arch == 'resnet34':
        model_data['model'] = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model_data['model'] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
    model_data['num_classes'] = len(model_data['class_names'])
    model_data['num_features'] = model_data['model'].fc.in_features
    model_data['model'].fc = nn.Linear(model_data['num_features'], model_data['num_classes'])
    if not silent:
        print(f"feature count: {model_data['num_features']}")
    
    model_data['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_data['model'] = model_data['model'].to(model_data['device'])
    if not silent:
        print(f"device: {model_data['device']}")
    
    model_data['criterion'] = nn.CrossEntropyLoss(label_smoothing)
    model_data['optimizer'] = torch.optim.Adam(model_data['model'].parameters(), lr=lr, weight_decay=weight_decay)
    model_data['scheduler'] = torch.optim.lr_scheduler.StepLR(model_data['optimizer'], step_size=7, gamma=0.1)

    return model_data
    
def prepare_for_retraining(model_data, train_dir, val_dir, batch_size=32, image_size=224, robustness=0.3, silent=False):
    model_data = prepare_dataloaders(model_data, train_dir, val_dir, batch_size, image_size, robustness, silent)

    new_classes_cnt = 0
    new_classes = []
    for class_name in model_data['datasets']['train'].classes:
        if class_name not in model_data['class_names']:
            model_data['class_names'].append(class_name)
            new_classes_cnt += 1
            new_classes.append(class_name)
    old_num_classes = model_data['num_classes']
    model_data['num_classes'] = len(model_data['class_names'])
    if not silent:
        print(f"{new_classes_cnt} new classes added: {new_classes}")
    
    old_fc_weights = model_data['model'].fc.weight.data[:old_num_classes]
    model_data['model'].fc = nn.Linear(model_data['num_features'], model_data['num_classes'])
    model_data['model'].fc.weight.data[:old_num_classes] = old_fc_weights
    if not silent:
        print(f"feature count: {model_data['num_features']}")
    
    model_data['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_data['model'] = model_data['model'].to(model_data['device'])
    if not silent:
        print(f"device: {model_data['device']}")
    
    return model_data;
    
def train(model_data, num_epochs, model_path, phases=['train', 'val'], break_at_val_acc_diff=None):
    start_time = time.time()
    last_val_acc = -1.0
    break_loop = False
    response = {}
    for epoch in range(num_epochs):
        print(f"Epoch {(epoch+1):4} / {num_epochs:4}", end=' ')
        for phase in phases:
            if phase == 'train':
                model_data['model'].train()
            else:
                model_data['model'].eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in model_data['dataloaders'][phase]:
                inputs, labels = inputs.to(model_data['device']), labels.to(model_data['device'])
                model_data['optimizer'].zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_data['model'](inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = model_data['criterion'](outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        model_data['optimizer'].step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(model_data['datasets'][phase])
            epoch_acc = running_corrects.double() / len(model_data['datasets'][phase])
            response[f"{phase}_loss"] = epoch_loss
            response[f"{phase}_acc"] = epoch_acc
            print(f" | {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}", end=' ')
            if phase == 'train':
                model_data['scheduler'].step()
            if phase == 'val':
                if break_at_val_acc_diff and epoch_acc - last_val_acc < break_at_val_acc_diff:
                    break_loop = True
                last_val_acc = epoch_acc
        print(f" | Elapsed time: {datetime.timedelta(seconds=(time.time() - start_time))}")
        torch.save(model_data, model_path.replace("###", f"{epoch:04}"))
        if break_loop:
            break;
    return response

def predict(image_path, model_data):
    model_data['model'].eval()
    image = Image.open(image_path).convert("RGB")
    image = model_data['transform']['val'](image).unsqueeze(0).to(model_data['device'])
    with torch.no_grad():
        outputs = model_data['model'](image)
        _, preds = torch.max(outputs, 1)
    try:
        return model_data['class_names'][preds[0]]
    except Exception:
        return None

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

def validate_prediction_in_dir(test_dir, model_data):
    model_data['model'].eval()
    total = 0
    success = 0
    failures = {}
    for species_dir in Path(test_dir).iterdir():
        if species_dir.is_dir():
            for file in Path(f"{species_dir}").iterdir():
                if file.is_file():
                    species = file.parts[-2]
                    prediction = predict(file, model_data)
                    is_success = (species==prediction)
                    if not is_success:
                        failures[species] = prediction
                    total = total + 1
                    if is_success:
                        success = success + 1
    return {
        'total': total, 
        'success': success,
        'failures': failures
    }

def test(model_data, test_dir, print_failures=True):
    model_data['model'].eval()
    start_time = time.time()
    prediction = validate_prediction_in_dir(test_dir, model_data)
    print(f"Accuracy: {prediction['success']} / {prediction['total']} -> {100*prediction['success']/prediction['total']:.2f}%")
    print(f"Elapsed time: {datetime.timedelta(seconds=(time.time() - start_time))}")
    if print_failures:
        print("-"*10)
        print("Failures:")
        pprint.pprint(prediction['failures'])

def test_top_k(model_data, test_dir, k, print_preds=True, print_accuracy=True, print_top1_accuracy=True, print_no_match=False, match_filter=0.0, print_genus_match=True):
    model_data['model'].eval()
    top1_success_cnt = 0
    top1_genus_success_cnt = 0
    success_cnt = 0
    genus_success_cnt = 0
    total_cnt = 0
    max_file_name_length = max([ len(file.name.split('.')[0]) for file in Path(test_dir).iterdir() ])
    for file in Path(test_dir).iterdir():
        if print_preds:
            print(f"{file.name.split('.')[0]:{max_file_name_length+1}}:", end=' ')
        total_cnt = total_cnt + 1
        probs = predict_top_k(file, model_data, k)
        genus_matched = False
        species_matched = False
        for pred, prob in probs.items():
            if not match_filter or prob >= match_filter:
                if pred in file.name:
                    species_matched = True
                    success_cnt = success_cnt + 1
                if pred.split('-')[0] in file.name:
                    genus_matched = True
                if print_preds and pred in file.name:
                    print(f"\033[32m{pred}\033[0m({prob:.3f}) ", end=' ')
                elif print_preds:
                    print(f"{pred}({prob:.3f}) ", end=' ')
        if genus_matched:
            genus_success_cnt = genus_success_cnt + 1
        if [pred for pred, prob in probs.items()][0] in file.name:
            top1_success_cnt = top1_success_cnt + 1
        if [pred.split('-')[0] for pred, prob in probs.items()][0] in file.name:
            top1_genus_success_cnt = top1_genus_success_cnt + 1
        if print_preds:
            print()
        if not print_preds and print_no_match and not species_matched:
            print(f"{file.name.split('.')[0]}:", end=' ')
            i = 0
            for pred, prob in probs.items():
                if i % 4 == 0:
                    print("\n\t", end=' ')
                if not match_filter or prob >= match_filter:
                    print(f"\033[{'33' if genus_matched else '31'}m{pred}\033[0m({prob:.3f}) ", end=' ')
                i += 1
            print()
    if print_accuracy:
        if print_preds:
            print("-"*10)
        if print_top1_accuracy:
            if print_genus_match:
                print(f"Top   1 accuracy: {top1_success_cnt}/{total_cnt} -> {100*top1_success_cnt/total_cnt:.2f}%, genus matched: {top1_genus_success_cnt}/{total_cnt} -> {100*top1_genus_success_cnt/total_cnt:.2f}%")
            else:
                print(f"Top   1 accuracy: {top1_success_cnt}/{total_cnt} -> {100*top1_success_cnt/total_cnt:.2f}%")
        if print_genus_match:
            print(f"Top {k:3} accuracy: {success_cnt}/{total_cnt} -> {100*success_cnt/total_cnt:.2f}%, genus matched: {genus_success_cnt}/{total_cnt} -> {100*genus_success_cnt/total_cnt:.2f}%")
        else:
            print(f"Top {k:3} accuracy: {success_cnt}/{total_cnt} -> {100*success_cnt/total_cnt:.2f}%")

def extract_proto_dataset(data_dir, proto_data_dir, limit):
    file_cnt = 0
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir() and os.listdir(class_dir):
            file_count = sum(1 for file in class_dir.iterdir() if file.is_file())
            class_dir_name = class_dir.name
            for file in Path(class_dir).iterdir():
                if file.is_file():
                    target_dir_path = f"{proto_data_dir}/{class_dir_name}"
                    if not os.path.exists(target_dir_path):
                        os.makedirs(target_dir_path)
                    shutil.copy(file, target_dir_path)
                    file_cnt = file_cnt + 1
                    if(file_cnt >= limit):
                        return

def class_count(data_dir, class_regex=None):
    class_cnt = 0
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir() and os.listdir(class_dir) and (not class_regex or re.match(class_regex, class_dir.name)):
            class_cnt += 1
    return class_cnt
    
def image_count(data_dir):
    img_cnt = 0
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            for file in Path(class_dir).iterdir():
                if file.is_file():
                    img_cnt += 1
    return img_cnt

def test_class(model_data, test_dir, classes):
    model_data['model'].eval()
    for expected_class in classes:
        total = 0
        success = 0
        for file in Path(f"{test_dir}/{expected_class}").iterdir():
            if file.is_file():
                prediction = predict(file, model_data)
                is_success = (expected_class==prediction)
                total = total + 1
                if is_success:
                    success = success + 1
        print(f"Class: {expected_class:15} ----> Success: {success}/{total} -> {100*success/total:.2f}%")


def analyze_top_k(image_path, model_data, k):
    model_data['model'].eval()
    image = Image.open(image_path).convert("RGB")
    image = model_data['transform']['val'](image).unsqueeze(0).to(model_data['device'])
    with torch.no_grad():
        outputs = model_data['model'](image)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k)
    try:
        return { model_data['class_names'][top_indices[0][i]]: {
            'class_idx': top_indices[0][i], 
            'class_name': model_data['class_names'][top_indices[0][i]], 
            'softmax': top_probs[0][i].item()} for i in range(0, k) }
    except Exception:
        return None

def test_class_for_min_accepted_softmax(model_data, test_dir, classes, min_accepted_softmax, k=0, print_success=True):
    if k <= 0 or k > len(classes):
        k = len(classes)
    model_data['model'].eval()
    full_success_cnt = 0
    for expected_class in classes:
        total = 0
        success = 0
        class_dir = f"{test_dir}/{expected_class}"
        if not os.path.exists(class_dir):
            continue;
        for file in Path(class_dir).iterdir():
            if file.is_file():
                predictions = analyze_top_k(file, model_data, k)
                is_success = expected_class in [ p['class_name'] for _,p in predictions.items() if p['softmax'] >= min_accepted_softmax ]
                total = total + 1
                if is_success:
                    success = success + 1
        if success/total == 1.0:
            full_success_cnt += 1
            if print_success:
                print(f"Class: {expected_class:15} ----> Success: {success}/{total} -> {100*success/total:.2f}%")
        else:
            print(f"\033[31mClass: {expected_class:15} ----> Success: {success}/{total} -> {100*success/total:.2f}%\033[0m")
    if not print_success:
        print(f"{full_success_cnt} classes ----> Success: 100%")



def plot_dist(data, start, end, step, xlabel, ylabel, title):
    bins = np.arange(start, end+step, step)
    plt.figure(figsize=(4, 3))
    sns.histplot(data, bins=bins, edgecolor='black', alpha=0.7, color='c')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

plot_data = Queue()
def run_plot_confidence_thread(model_data, test_dir, expected_class, min_accepted_softmax, k):
    total = 0
    success = 0
    class_dir = f"{test_dir}/{expected_class}"
    if not os.path.exists(class_dir):
        return;
    for file in Path(class_dir).iterdir():
        if file.is_file():
            predictions = analyze_top_k(file, model_data, k)
            is_success = expected_class in [ p['class_name'] for _,p in predictions.items() if p['softmax'] >= min_accepted_softmax ]
            total = total + 1
            if is_success:
                success = success + 1
    global plot_data
    plot_data.put(100*success/total)
def plot_confidence(model_data, test_dir, classes, min_accepted_softmax, k=0):
    if k <= 0 or k > len(classes):
        k = len(classes)
    model_data['model'].eval()
    global plot_data
    plot_data = Queue()
    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for expected_class in classes:
            futures += [executor.submit(run_plot_confidence_thread, model_data, test_dir, expected_class, min_accepted_softmax, k)]
    for future in futures:
        future.result()
    plot_dist(list(plot_data.queue), 0, 100, 2, f"top-{k} success on >={100*min_accepted_softmax}% confidence", "class count", test_dir)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
