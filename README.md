# ELEC475-ComputerVision

## Overview of Computer Vision Labs Repository

This repository features a collection of hands-on computer vision projects from labs in ELEC475 @ Queen's university. The labs progress from classic tasks like image reconstruction and localization to advanced segmentation with custom neural networks and knowledge distillation. Below is a summary of the main projects:

---

### 1. MNIST Autoencoder - Lab 1

**Description:**  
Implements a 4-layer Multi-Layer Perceptron (MLP) autoencoder on the MNIST handwritten digits dataset. The model compresses images into a low-dimensional bottleneck and reconstructs them, with a focus on handling noisy inputs.

**Features:**
- **Autoencoding:** Reconstructs input digits, generally preserving digit shape.
- **Denoising:** Removes Gaussian noise; some reconstructions may morph digits into similar-looking characters.
- **Interpolation:** Performs linear interpolation in the bottleneck space to visualize smooth transitions between digits.

**Dataset:**  
MNIST

---

### 2. Pet Nose Localization - SnoutNet - Lab 2

**Description:**  
Develops a convolutional neural network, SnoutNet, to predict the (u, v) coordinates of pet noses in images.

**Features:**
- **Model Architecture:** 3 convolutional layers followed by 3 fully connected layers.
- **Data Augmentation:** Uses horizontal flip, 90-degree rotation, and their combination to expand the dataset.
- **Performance Evaluation:** Assesses mean and maximum localization errors for each augmentation strategy.

**Dataset:**  
Custom-labeled pet nose dataset. Courtesy of Prof. Greenspan

---

### 3. Ensemble Image Classification

**Description:**  
Applies ensemble learning to the CIFAR-100 dataset using three pre-trained models (VGG-16, AlexNet, ResNet-18) with ensemble methods to improve classification accuracy.

**Features:**
- **Pre-trained Models:** Adapts final layers for CIFAR-100.
- **Training:** Uses Adam optimizer, ReduceLROnPlateau scheduler, and cross-entropy loss.
- **Ensemble Methods:** Compares maximum probability, average probability, and majority voting.
- **Performance:** Maximum probability ensemble achieves the lowest error rate after convergence.

**Dataset:**  
CIFAR-100 (32x32 color images, 100 classes).

---

### 4. Custom Semantic Segmentation Model (DMadNet) - Lab 4&5

**Description:**  
This project centers on semantic segmentation using a custom deep neural network, DMadNet, inspired by ResNet-18 as an encoder and a multi-stage decoder with skip connections. The model is trained and evaluated on the PASCAL VOC 2012 dataset for pixel-wise classification.

**Features:**
- **Model Architecture:**  
  - Encoder based on ResNet-18, leveraging pre-trained weights for feature extraction.
  - Decoder uses multiple blocks with skip connections, upsampling, and dropout for robust segmentation.
  - Final output is upsampled to match the original input size.
- **Training Pipeline:**  
  - Supports advanced data augmentation (random flips, color jitter, normalization) for improved generalization.
  - Implements early stopping and learning rate scheduling to prevent overfitting and optimize convergence.
  - Uses weighted cross-entropy loss to address class imbalance in segmentation masks.
- **Knowledge Distillation (Optional):**  
  - Includes a teacher-student framework where DMadNet (student) learns from a larger pre-trained model (ResNet-50 in our case).
  - Supports both response-based and feature-based distillation loss.

**Dataset:**  
PASCAL VOC 2012 (21 semantic classes, 256x256 images after resizing and augmentation).

