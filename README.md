# Speech Emotion Recognition with Spiking Neural Networks

## Overview
[cite_start]This repository contains the implementation of a hybrid Speech Emotion Recognition (SER) framework that combines Convolutional Neural Networks (CNNs) with Spiking Neural Networks (SNNs)[cite: 9]. [cite_start]Traditional SER systems relying solely on deep learning models often require high computational power, limiting their deployment on edge devices[cite: 7]. [cite_start]By integrating CNN-based feature extraction with an SNN back-end utilizing Leaky Integrate-and-Fire (LIF) neurons, this project offers an energy-efficient, biologically plausible alternative suitable for neuromorphic hardware platforms[cite: 8, 13, 60].

## Authors
* [cite_start]**Le Hoang Viet** - University of Information Technology [cite: 3, 4]
* [cite_start]**Hoang Cong Chien** - University of Information Technology [cite: 5]

## Datasets
[cite_start]The model is evaluated on a harmonized dataset constructed from four benchmark emotional speech corpora[cite: 12]:
* [cite_start]**Sources**: RAVDESS, TESS, CREMA-D, and SAVEE[cite: 12].
* [cite_start]**Emotion Classes**: Data is standardized into five categories: Neutral, Happy, Sad, Angry, and Fearful[cite: 92, 93, 94, 95].
* [cite_start]**Volume**: The final dataset comprises 10,580 balanced audio samples[cite: 110, 111]. [cite_start]All audio is resampled to 16kHz, amplitude-normalized, and trimmed/padded to a uniform 3-second duration[cite: 100, 101, 102].

## Feature Engineering
[cite_start]The system utilizes a multi-dimensional acoustic feature representation to capture both spectral and prosodic characteristics[cite: 115]:
* [cite_start]**Stacked Features**: Frame-wise concatenation of 13 Mel-Frequency Cepstral Coefficients (MFCCs), 1 Root Mean Square Energy (RMSE) descriptor, and 1 Zero-Crossing Rate (ZCR) descriptor[cite: 202].
* [cite_start]**Input Shape**: The extraction pipeline yields a 15 x 299 feature matrix per audio clip[cite: 205].

## Model Architecture
[cite_start]The hybrid architecture is structured into two main stages[cite: 402]:

1. [cite_start]**CNN Front-end**: Three sequential convolutional blocks[cite: 406]. [cite_start]Each block applies a 2D Convolution (3x3 kernel), Batch Normalization, ReLU activation, and 2x2 Max Pooling to extract localized spatio-temporal patterns[cite: 407, 432, 451, 477]. [cite_start]The output is flattened into a 128-dimensional embedding[cite: 497].
2. [cite_start]**Spike Encoding**: Continuous-valued features are converted into spike trains using Rate Coding[cite: 270, 499].
3. [cite_start]**SNN Back-end**: A three-layer fully connected spiking network using LIF neurons, structured as 128 -> 64 -> 32 -> 5 neurons[cite: 501, 506, 536, 560, 562]. 

## Training Methodology
[cite_start]Training SNNs from scratch presents challenges due to the non-differentiable nature of spike generation[cite: 597]. This project employs:
* [cite_start]**Surrogate Gradient Descent**: Approximating the spike function derivative using a Fast Sigmoid Surrogate to enable backward passes[cite: 620, 623].
* [cite_start]**Optimization**: Backpropagation Through Time (BPTT) coupled with the Adam optimizer[cite: 654, 804].

## Key Results
* [cite_start]**Feature Superiority**: The stacked feature representation (MFCC + RMSE + ZCR) achieved 71.9% baseline accuracy, significantly outperforming the MFCC mean approach (63.3%)[cite: 809, 814, 815].
* [cite_start]**Classification Accuracy**: The fine-tuned SNN model achieved an overall accuracy of 87.8%[cite: 821]. [cite_start]This is within 0.5% of the baseline CNN performance while drastically reducing floating-point operations[cite: 821, 823].

## Installation

[cite_start]The model is implemented in PyTorch using the `snnTorch` library[cite: 804]. 

```bash
# Clone the repository
git clone https://github.com/rivaille1704/Speech-Emotion-Recognition-with-Spiking-Neural-Network-.git
cd Speech-Emotion-Recognition-with-Spiking-Neural-Network-

# Install required dependencies
pip install torch torchvision torchaudio
pip install snntorch numpy pandas scikit-learn librosa
