# Session 5 PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for training and testing on the MNIST dataset. The code includes data loading, model definition, training loop, and evaluation.

## Getting Started

### Prerequisites

- Python 3
- PyTorch
- torchvision
- matplotlib
- tqdm

### Training the Model
Clone the repository and run the training script:<br>
```
python S5_Himank_Jain.ipynb

```
### Code Structure
The code is divided into thress files:
- S5_Himank_Jain.ipynb: Main script for training the CNN on the MNIST dataset. Here we load the dataset, visualise some samples, and train the model (from model.py) using functions from utils.py
- model.py: Definition of the CNN architecture. This file contains two models:
  - Net(): CNN architecture with bias
  - Net2(): CNN architecture without bias
- utils.py: Utility functions for transformations, and training/testing.
