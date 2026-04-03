# Speech Emotion Recognition with Spiking Neural Networks

## Overview
This repository contains the implementation of a hybrid Speech Emotion Recognition (SER) framework that combines Convolutional Neural Networks (CNNs) with Spiking Neural Networks (SNNs). Traditional SER systems relying solely on deep learning models often require high computational power, limiting their deployment on edge devices. By integrating CNN-based feature extraction with an SNN back-end utilizing Leaky Integrate-and-Fire (LIF) neurons, this project offers an energy-efficient, biologically plausible alternative suitable for neuromorphic hardware platforms.

## Authors
* **Le Hoang Viet** - University of Information Technology
* **Hoang Cong Chien** - University of Information Technology

## Datasets
The model is evaluated on a harmonized dataset constructed from four benchmark emotional speech corpora:
* **Sources**: RAVDESS, TESS, CREMA-D, and SAVEE.
* **Emotion Classes**: Data is standardized into five categories: Neutral, Happy, Sad, Angry, and Fearful.
* **Volume**: The final dataset comprises 10,580 balanced audio samples. All audio is resampled to 16kHz, amplitude-normalized, and trimmed/padded to a uniform 3-second duration.

## Feature Engineering
The system utilizes a multi-dimensional acoustic feature representation to capture both spectral and prosodic characteristics:
* **Stacked Features**: Frame-wise concatenation of 13 Mel-Frequency Cepstral Coefficients (MFCCs), 1 Root Mean Square Energy (RMSE) descriptor, and 1 Zero-Crossing Rate (ZCR) descriptor.
* **Input Shape**: The extraction pipeline yields a 15 x 299 feature matrix per audio clip.

## Model Architecture
The hybrid architecture is structured into two main stages:

1. **CNN Front-end**: Three sequential convolutional blocks. Each block applies a 2D Convolution (3x3 kernel), Batch Normalization, ReLU activation, and 2x2 Max Pooling to extract localized spatio-temporal patterns. The output is flattened into a 128-dimensional embedding.
2. **Spike Encoding**: Continuous-valued features are converted into spike trains using Rate Coding.
3. **SNN Back-end**: A three-layer fully connected spiking network using LIF neurons, structured as 128 -> 64 -> 32 -> 5 neurons. 

## Training Methodology
Training SNNs from scratch presents challenges due to the non-differentiable nature of spike generation. This project employs:
* **Surrogate Gradient Descent**: Approximating the spike function derivative using a Fast Sigmoid Surrogate to enable backward passes.
* **Optimization**: Backpropagation Through Time (BPTT) coupled with the Adam optimizer.

## Key Results
* **Feature Superiority**: The stacked feature representation (MFCC + RMSE + ZCR) achieved 71.9% baseline accuracy, significantly outperforming the MFCC mean approach (63.3%).
* **Classification Accuracy**: The fine-tuned SNN model achieved an overall accuracy of 87.8%. This is within 0.5% of the baseline CNN performance while drastically reducing floating-point operations.

## Installation
The model is implemented in PyTorch using the `snnTorch` library. 

```bash
# Clone the repository
git clone [https://github.com/rivaille1704/Speech-Emotion-Recognition-with-Spiking-Neural-Network-.git](https://github.com/rivaille1704/Speech-Emotion-Recognition-with-Spiking-Neural-Network-.git)
cd Speech-Emotion-Recognition-with-Spiking-Neural-Network-

# Install required dependencies
pip install torch torchvision torchaudio
pip install snntorch numpy pandas scikit-learn librosa
