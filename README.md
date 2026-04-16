# Disaster Image Classification and Severity Analysis

## Overview
This project focuses on building a deep learning pipeline for disaster image classification using a convolutional neural network. The system classifies disaster types and incorporates a secondary task for severity estimation using a multi-task learning approach.

The workflow includes data preprocessing, handling class imbalance, model training, evaluation, and interpretability using Grad-CAM.

---

## Dataset
- Format: Image dataset organized in class-based folders
- Task:
  - Primary: Disaster type classification
  - Secondary: Severity prediction (synthetic labels used)

---


### 1. Environment Setup
- Framework: PyTorch
- GPU support enabled if available
- Key libraries:
  - torch, torchvision
  - numpy, pandas
  - matplotlib, seaborn
  - PIL, OpenCV

---

### 2. Data Loading
- Dataset loaded using `ImageFolder`
- Directory structure assumed:
  ```
  root/
    class_1/
    class_2/
    ...
  ```

- Extracted:
  - Class labels
  - Number of classes

---

### 3. Data Preprocessing

#### Training Transformations
- Resize to 256x256
- Random resized crop (224x224)
- Horizontal and vertical flips
- Normalization (ImageNet mean/std)

#### Validation Transformations
- Resize to 224x224
- Normalization

---

### 4. Train-Validation Split
- 80% training data
- 20% validation data
- Performed using `random_split`

---

### 5. Class Imbalance Handling
- Computed class distribution using label counts
- Applied inverse frequency weighting
- Used `WeightedRandomSampler` to balance training batches

---

### 6. Data Loaders
- Training loader:
  - Uses weighted sampler
  - Batch size: 32

- Validation loader:
  - No shuffling
  - Batch size: 32

---

## Model Architecture

### Backbone
- Pretrained `ResNet50` (ImageNet weights)

### Multi-Task Design
- Shared convolutional backbone
- Two output heads:
  1. Disaster classification head
  2. Severity prediction head

### Key Idea
Multi-task learning allows the model to learn shared representations that improve generalization.

---


### Hyperparameters
- Epochs: 5
- Learning rate: 1e-4
- Optimizer: Adam
- Loss functions:
  - CrossEntropyLoss (classification)
  - CrossEntropyLoss (severity)

## What this project does
This project provides a small web app and an API that analyze an uploaded image and predict the type of disaster in the scene. It also estimates how severe the damage looks inside the predicted class using a Grad-CAM based score and simple visual heuristics.

## Main features
- Image classification into six classes: Damaged_Infrastructure, Fire_Disaster, Human_Damage, Land_Disaster, Non_Damage, Water_Disaster
- Intra-class severity tiering: Contained, Moderate, Severe, Catastrophic
- FastAPI backend with a single /predict endpoint for image inference
- Simple web interface for drag-and-drop image upload

## Tech stack
- Python, FastAPI, Uvicorn
- PyTorch, Torchvision
- OpenCV, Pillow
- HTML, CSS, JavaScript

## Repository contents
- app.py: FastAPI server, model loading, inference, and severity scoring
- best_model.pth: Trained model weights
- index1.html, script1.js, style2.css: Front-end UI
- requirements.txt: Python dependencies
- syntaxError_Phase1.ipynb: Training and experimentation notebook
- syntax_error_ppt.pdf: Project presentation

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt

## Use the web interface
1. Open index1.html in your browser.
2. Upload an image.
3. Click Analyze Image.

Note: The front-end currently calls the API at http://127.0.0.1:8000/predict. Update script1.js if your API runs on a different host or port.

## API endpoint
POST /predict
- Request: multipart/form-data with a single file field named file
- Response: predicted_class, confidence, intra_class_severity, intra_severity_score, intra_severity_rationale, all_scores

## Model and severity logic
The model is a ResNet-50 classifier. Intra-class severity is computed using a weighted mix of:
- Grad-CAM activation spread
- Class-specific visual proxy scores (fire, water, land, infrastructure, human)
- Confidence margin between the top predictions

## Notes
- This repository contains a large model file (best_model.pth). Make sure you have enough disk space when cloning.
- Use JPEG or PNG images for inference.