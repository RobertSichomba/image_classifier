# MNIST Image Classifier

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The model achieves high accuracy and can be used to predict digits from new images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Making Predictions](#making-predictions)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Epoch Choice](#epoch-choice)
9. [Future Improvements](#future-improvements)
10. [License](#license)

---

## Project Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9). This project uses a CNN to classify these images with high accuracy. The model is built using **TensorFlow** and **Keras**, and it achieves a test accuracy of over **99%**.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow (PIL)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RobertSichomba/image_classifier.git
   cd image_classifier
