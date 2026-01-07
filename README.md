Fake Image Detection System

A deep learning–based system for detecting real vs AI-generated (fake) images using transfer learning with ResNet18.
Built with a focus on robust preprocessing, generalization, and reproducible ML workflows.

Overview

With the rise of AI-generated media, fake images pose serious risks such as misinformation, fraud, and identity misuse.
This project addresses the problem by fine-tuning a pre-trained ResNet18 CNN to classify images as Real or Fake.

The system integrates:

Structured dataset preprocessing

Data augmentation for robustness

Transfer learning for efficient training

Clear evaluation on unseen data

Key Features

Binary image classification: Real vs Fake

Transfer learning using ResNet18 (ImageNet pre-trained)

Robust preprocessing (normalization, resizing, safe image loading)

Data augmentation to improve generalization

High validation accuracy with minimal overfitting

Tech Stack

Language: Python

Framework: PyTorch, Torchvision

Model: ResNet18 (Transfer Learning)

Libraries: NumPy, Pandas, Matplotlib

Tools: Jupyter Notebook, VS Code

Dataset

The dataset contains labeled images split into two classes:

Class 0: Real images

Class 1: Fake (AI-manipulated) images

Distribution
Dataset	Real	Fake
Training	23,999	23,998
Testing	6,781	7,210

Images vary in resolution, lighting, and content to improve real-world generalization.

Preprocessing & Augmentation

Removal of corrupted or unsupported files

Resizing all images to 512 × 512

Normalization using ImageNet mean & std

Training-only augmentation:

Random horizontal flip

Color jitter (brightness, contrast, saturation)

These steps prevent shortcut learning (e.g., image size bias) and improve robustness.

Model Architecture

Base model: ResNet18 (pre-trained on ImageNet)

Final fully connected layer modified for 2 classes

Loss function: CrossEntropyLoss

Optimizer: Adam (lr = 1e-4)

Epochs: 5

Transfer learning significantly reduces training time while maintaining strong performance.

Results
Metric	Training	Validation
Accuracy	96–97%	~97%
Loss (final)	~0.03	~0.04

Strong convergence without overfitting

Balanced precision across both classes

Fake images are slightly harder to detect due to subtle artifacts

Applications

Digital forensics

Social media content moderation

Image authenticity verification systems

Research on AI-generated media detection

Limitations

Trained only on still images (no video/audio)

Performance may drop on unseen DeepFake generation styles

No real-time inference optimization yet

Future Improvements

Extend to video DeepFake detection

Add model explainability (Grad-CAM)

Train with more diverse DeepFake sources

Optimize inference for real-time usage

Experiment with EfficientNet or Vision Transformers

Project Structure (simplified)
Fake-image-detection-system/
│── data/
│   ├── train/
│   ├── test/
│── notebooks/
│── models/
│── README.md

Author

Dhanani Iyan
MCA Student | Full-Stack & ML Developer
GitHub: https://github.com/iyan-devcore
