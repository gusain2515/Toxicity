# Toxic Comment Classification with TensorFlow

A bidirectional LSTM-based model to identify toxic comments (toxic, severe_toxic, obscene, threat, insult, identity_hate) using the Jigsaw Toxic Comment Classification dataset. This repository includes data preprocessing, model training, evaluation, and a simple Gradio interface for live inference.

---

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Deployment (Gradio Interface)](#deployment-gradio-interface)  
- [Usage](#usage)  
- [License](#license)  

---

## Features

- **Data Loading & Preprocessing**  
  - Reads the Jigsaw CSV file  
  - Text vectorization using `TextVectorization` layer  
  - Creates TensorFlow Dataset with caching, shuffling, batching, and prefetching  

- **Model Architecture**  
  - Embedding layer (vocabulary size: 200,000 + 1, embedding dimension: 32)  
  - Bidirectional LSTM (32 units, tanh activation)  
  - Three fully connected layers (128 → 256 → 128 units, ReLU)  
  - Output layer with 6 sigmoid units (one per toxicity label)  

- **Training & Validation**  
  - Binary cross-entropy loss  
  - Adam optimizer  
  - One epoch of training by default (can be adjusted)  
  - Validation split  

- **Evaluation Metrics**  
  - Precision  
  - Recall  
  - Categorical Accuracy  

- **Live Inference (Gradio)**  
  - Simple web interface to type in a comment and get toxicity predictions  

---

## Prerequisites

- Python 3.8+  
- pip (Python package installer)  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/toxic-comment-classification.git
   cd toxic-comment-classification
