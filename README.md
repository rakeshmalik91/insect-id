# Insect Species Identification

<div align="center">
	<img src="insect-id-app/logo.png" alt="InsectID-AI Logo" width="150"/>
	<br/>
	<b>AI-Powered Identification for Indian Insects</b>
</div>

<br/>

**InsectID** is an open-source project combining deep learning and mobile technology to identify insect species from images. With a focus on **Indian** biodiversity, the system is currently trained to recognize thousands of species including **butterflies, moths, dragonflies, and cicadas**.

> ⚠️ **Note**: This project is currently in an **early prototype stage**. Mechanics, visuals, and features are subject to change and improvement.
>
> 🛑 **Disclaimer**: This project is developed for **educational purposes and personal use only**.

## 📱 Mobile App (Android)

Identify insects on the go with the Android application.

- **[Google Play Store](https://play.google.com/store/apps/details?id=com.rakeshmalik.insectid)** (Latest Beta)
- **APK Archives**: [Google Drive](https://drive.google.com/drive/folders/1UNogisKp3rtcOnigcibAPiNsQB-gZJpD?usp=drive_link) | [Dropbox](https://www.dropbox.com/scl/fo/5t1zgkn419ctlzkuacu3h/ACC-_MbfOOu151yPRRH25XU?rlkey=3tirqkq5xland2qx3dfa8hrda&st=0aosjy2b&dl=0)

<p align="center">
	<img src="insect-id-app/screenshots/0.jpg" alt="Screenshot" width="130" style="margin: 5px;"/>
	<img src="insect-id-app/screenshots/1.jpg" alt="Screenshot" width="130" style="margin: 5px;"/>
	<img src="insect-id-app/screenshots/2.jpg" alt="Screenshot" width="130" style="margin: 5px;"/>
	<img src="insect-id-app/screenshots/4.jpg" alt="Screenshot" width="130" style="margin: 5px;"/>
	<img src="insect-id-app/screenshots/5.jpg" alt="Screenshot" width="130" style="margin: 5px;"/>
</p>

## 🚀 How to Use

1. **Launch the App**: Open InsectID on your Android device.
2. **Manage Models**:
   - Before identifying, ensure you download the relevant model for your insect type.
   - Available Models: **Butterfly & Moth**, **Dragonfly & Damselfly**, **Cicada**.
   - *Note: Models require internet to download once, but identification works **completely offline** afterwards.*
3. **Choose Input Method**:
   - 📷 **Camera**: Point your camera at an insect to capture a photo.
   - 🖼️ **Gallery**: Select an existing image from your photo library.
4. **Crop & Focus**: For best results, crop the image so the insect fills most of the frame.
5. **Select Model & Identify**: Choose the correct model group from the dropdown/menu and tap the checkmark to analyze.
6. **Results**: The AI will list the most likely species matches with confidence scores.

## 🧠 Model & Data Statistics

The current model is trained on **378k+ images** covering **4,767 unique species/classes**.

| Species Group | Covered (Count) | Estimated (India) |
| :--- | :--- | :--- |
| **Moths** (Lepidoptera) | 2,899 | 12,000+ |
| **Butterflies** (Lepidoptera) | 1,501 | 1,300+ |
| **Dragonflies/Damselflies** (Odonata) | 510 | 760+ |
| **Cicadas** (Hemiptera) | 300 | 250+ |

### Dataset Sources
| Source | Images | Classes | Region | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [Moths of India](https://www.mothsofindia.org) | 44k | 3,364 | India | Primary source for moths |
| [Butterflies of India](https://www.ifoundbutterflies.org) | 66k | 1,554 | India | High quality verified images |
| [Indian Odonata](https://www.indianodonata.org) | 13k | 737 | India | Includes empty classes |
| [Indian Cicadas](https://www.indiancicadas.org) | 1k | 308 | India | Sparse data |
| [iNaturalist](https://www.inaturalist.org) | 232k | 4,221 | India | Large volume, mixed quality |
| [India Biodiversity](https://indiabiodiversity.org) | 12k | 1,444 | India | Legacy names, some typos |
| [Insecta.pro](https://insecta.pro) | 25k | 5,068 | Global | Low res images |
| [Wikipedia](https://www.wikipedia.org) | 2k | 1,825 | India | Reference only |

**Model Checkpoints**: [Google Drive](https://drive.google.com/drive/folders/1FtGjLJc_JNwLs0cey3euyzUxwpids10G?usp=drive_link)
**Raw Datasets**: [Google Drive](https://drive.google.com/drive/folders/10qLVcGkJlLplKjIluRc9GEyQhcqpyhhD?usp=drive_link)

## 🖥️ Command Line Tools

The project provides three main scripts for the end-to-end ML workflow: scraping, training, and testing.

### 1. Scraping Data (`scrape.py`)
Collects images from various online sources (e.g., MothsofIndia, iNaturalist) based on a species list (`species.json`).

```bash
# Scrape specific types (default: moth)
python scrape.py --types moth butterfly

# Scrape only new/missing species (skip existing directories)
python scrape.py --new-species
```

**Logs**: `logs/scrape.{type}.log`

### 2. Training (`train.py`)
Handles data aggregation, validation, and model training (from scratch or incremental).

```bash
# Default training (Lepidoptera, Version 1)
python train.py

# Train a specific version (e.g., experimental run)
python train.py -v v2 --max-epochs 50

# Resume training automatically (detects latest checkpoint)
python train.py -v v2

# Advanced: Skip data processing steps for faster startup
python train.py -v v2 --skip-aggregate --skip-validate

# Train on a different model type
python train.py -m odonata
```

**Logs**: `logs/train.{model_name}.{version}.log`

### 3. Evaluation (`test.py`)
Evaluates trained models on test datasets. Supports wildcards for batch testing.

```bash
# Test the latest checkpoint (auto-detected)
python test.py

# Test specific version and epoch
python test.py -v v2 -e 20

# Test on multiple directories (supports wildcards)
python test.py --test-dirs insect-dataset/src/test*/lepidoptera

# Detailed output (show predictions per image)
python test.py --print-preds --top-k 1 3 5
```

**Logs**: `logs/test.{model_name}.{version}.log`

### 4. Asset Publishing (`publish.py`)
Generates production assets (TorchScript models, image archives, metadata) from trained checkpoints.

```bash
# Publish all assets for a model
python publish.py -m lepidoptera -v v2

# Publish specific assets
python publish.py -m odonata --task images
python publish.py -m cicada --task model

# Publish ExecutionTorch model
python publish.py -m lepidoptera -v v2 --task model --executorch
```

**Logs**: `logs/publish.{model_name}.{version}.log`

## 🛠️ Python Library Usage (`mynnlib`)

The project includes custom wrapper libraries `mynnlib` and `mynnlibv2` for easy training and inference.

### Installation
Ensure you have `torch` (with CUDA support if available) and other dependencies installed.

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` includes an extra index URL for PyTorch with CUDA 11.8 support. Adjust if necessary for your system.*

### Inference Example
```python
import mynnlib
from mynnlib import predict, predict_top_k

# Load Model
model_data = torch.load("path/to/checkpoint.pth", weights_only=False)

# Single Prediction
result = predict("image.jpg", model_data)
print(f"Species: {result}")

# Top-5 Predictions
top5 = predict_top_k("image.jpg", model_data, 5)
print(top5)
```

### Training Example (Incremental)
Use `mynnlibv2` for incremental learning capabilities.

```python
import mynnlibv2
from mynnlibv2 import init_model, run_epoch

# Initialize New Model
model_data = init_model(
    train_dir="dataset/data",
    val_dir="dataset/val",
    batch_size=32,
    image_size=224,
    lr=1e-4
)

# Train Loop
for epoch in range(15):
    run_epoch(model_data, output_path="checkpoints/model", robustness_lambda=0.1)
```

## 🗓️ Backlog & Roadmap
- [ ] **Species Expansion**: Cover Beeltes, Hymenoptera (Bees/Wasps), and Orthoptera.
- [ ] **Lifecycle Handling**: Better classification for Larvae, Pupae, and Eggs.
- [ ] **App Improvements**:
    - Better screen capture handling.
    - User-controlled model downloading (move away from Google Drive).
- [ ] **Data Cleanup**: Resolve taxonomic synonyms and typos in source data.
- [ ] **Root Classifier**: Hierarchical model to first identify Order/Family before Species.

## 📚 Related Resources
- **Blog**: [Fixing libc++_shared.so conflicts on Android with PyTorch/OpenCV](https://medium.com/@ghostknife/libc-shared-so-conflicts-on-android-project-when-using-pytorch-and-opencv-811abb6322e6)
- **Issue**: [KB pagination support for PyTorch Lite](https://github.com/pytorch/pytorch/issues/154449)
