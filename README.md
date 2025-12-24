# 🏥 Lightweight Radiology Visual Question Answering (VQA) in Radiology Domain

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://pang-lang-datascienceproject-radvqa-streamlit-r3iiyd.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/daphne04/radvqa-lightweight)
[![Canva Slides](https://img.shields.io/badge/Canva-Presentation-blue)](https://www.canva.com/design/DAG6g3MSKko/zi79GjGjoqmH82MbFdPpbA/view)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This project implements a **Lightweight Visual Question Answering (VQA)** system designed for radiology images.  
The system supports:

- **Binary questions** (yes/no), e.g. *“Is there a fracture?”*
- **Open-ended questions**, e.g. *“What organ is shown?”*

The goal is to study the **trade-off between accuracy and efficiency** for deployable medical VQA systems.

⚠️ *This is a research prototype and not intended for direct clinical use.*

---

## ✨ Key Features

- Supports radiology images (X-ray, CT, MRI)
- Attention-based multimodal feature fusion
- Dual-head architecture for binary and open-ended questions
- Lightweight model optimized for faster inference
- Interactive Streamlit web application

---

## 🏗️ Model Architecture

### Encoders
- **Vision Encoder**
  - Baseline: ResNet-34
  - Lightweight: MobileNetV3-Small
- **Text Encoder**
  - Baseline: BERT-base
  - Lightweight: DistilBERT

### Fusion
- Attention-based multimodal feature fusion

### Prediction Heads
- Binary head (yes / no)
- Open-ended head (medical terms)

### Model Variants

| Model | Vision Encoder | Text Encoder | Parameters | Size |
|------|---------------|--------------|------------|------|
| **Lightweight** | MobileNetV3-Small | DistilBERT | ~70M | ~270 MB |
| **Baseline** | ResNet-34 | BERT-base | ~134M | ~512 MB |

---

## 🎬 Demo

### Live Streamlit App
🔗 **Radiology VQA Demo**  
https://pang-lang-datascienceproject-radvqa-streamlit-r3iiyd.streamlit.app/

### Demo Capabilities
- Choose between lightweight and baseline models
- Upload radiology images (X-ray, CT, MRI)
- Ask natural language questions
- View predicted answers with confidence scores
- Display top-ranked predictions
- Visualize attention-based feature interaction

---

## 📊 Results Summary

- Lightweight model achieves comparable accuracy to the baseline
- Approximately **1.9× smaller model size** and **~1.04× faster inference**
- Suitable for edge and real-time deployment

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, for training)
- 8GB+ RAM

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/radiology-vqa.git
cd radiology-vqa
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Models
Models are automatically downloaded from HuggingFace when running the Streamlit app.
For manual download:
```bash
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="daphne04/radvqa-lightweight",
    filename="lightweight_best_model.pt"
)

hf_hub_download(
    repo_id="daphne04/radvqa-lightweight",
    filename="baseline_best_model.pt"
)
```

## 📁 Project Structure

```
radiology-vqa/
├── .streamlit/
│   └── config.toml                 # Streamlit UI configuration
│
├── analysis/
│   ├── augmentation/               # Augmentation visual checks
│   ├── benchmark_inference/        # Inference speed analysis
│   └── eda_reports/                # Exploratory data analysis outputs
│
├── evaluation/
│   ├── evaluate_dual_head.py       # Binary & open-ended evaluation
│   └── evaluate_roc.py             # AUC-ROC evaluation
│
├── evaluation_results/
│   ├── baseline/                   # Baseline evaluation metrics
│   ├── baseline_roc/               # Baseline ROC results
│   ├── lightweight/                # Lightweight evaluation metrics
│   └── lightweight_roc/            # Lightweight ROC results
│
├── models/
│   ├── lightweight_model.py        # Lightweight dual-head VQA model
│   └── baseline_model.py           # Baseline dual-head VQA model
│
├── preprocessing/
│   ├── check_unk.py                # UNK rate analysis
│   ├── combined_preprocessing.py   # Multimodal preprocessing pipeline
│   ├── image_preprocessing.py      # Image transforms & augmentation
│   ├── text_preprocessing.py       # Text tokenization & normalization
│   └── load_dataset.py             # Dataset loading & EDA
│
├── training/
│   ├── train_lightweight.py        # Train lightweight model
│   └── train_baseline.py           # Train baseline model
│
├── radvqa_streamlit.py             # Streamlit application
├── run_streamlit.sh                # Streamlit launch script
├── answer_vocab.json               # Final answer vocabulary
├── save_answer_vocab.py            # Vocabulary generation script
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
└── README.md                       # Project documentation             
```

---


<div align="center">

**Made with ❤️**

⭐ Star this repo if you find it helpful!

</div>

