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





