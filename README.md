# **🚀 TransUNet - Unofficial Implementation of Transformer-Based U-Net for Medical Image Segmentation**

This is an **unofficial implementation** of **[TransUNet](https://arxiv.org/abs/2102.04306)**:  
> **"Transformers Make Strong Encoders for Medical Image Segmentation"**  
> *Jieneng Chen, Yutong Lu, Qihang Yu, et al.*

TransUNet is a **hybrid deep learning model** that combines **CNNs (Convolutional Neural Networks)** and **Transformers** to improve segmentation accuracy in medical imaging. This repository provides a **fully configurable** and **easy-to-use** implementation of TransUNet.

---

## **📌 Key Features**
✅ **Unofficial Implementation** based on the **original paper**  
✅ Hybrid **CNN + Transformer** architecture for enhanced segmentation  
✅ Supports multiple optimizers: **Adam, AdamW, SGD**  
✅ Configurable loss functions: **BCE, Focal, Tversky**  
✅ Multi-device support: **CPU, GPU, Apple M1/M2 (`mps`)**  
✅ Automatic **dataset handling, model training, and evaluation**  
✅ Saves **segmentation masks** and **best-performing model checkpoints**  

---

## **📌 Model Architecture**
The **TransUNet** model follows a two-stage approach:  
1️⃣ **CNN Encoder** - Extracts local spatial features  
2️⃣ **Transformer Block** - Captures **global dependencies**  
3️⃣ **Decoder** - Combines CNN and Transformer outputs for segmentation  

```
Input Image (H x W x C)
       │
       ▼
█████ CNN Encoder █████   (Extracts feature maps)
       │
       ▼
-----> Transformer Block ----->   (Captures long-range dependencies)
       │
       ▼
█████ Decoder (UpSampling + Skip Connections) █████
       │
       ▼
█████ Segmentation Mask (Output) █████
```

---

## **📌 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/atikul-islam-sajib/TransUNet.git
cd TransUNet
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ (Optional) Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## **📌 Configuration - `config.yaml`**
Before running training or testing, modify `config.yaml` to set paths, model parameters, and training options.

```yaml
# 📌 Configuration File for TransUNet
# This file defines paths, data settings, model architecture, training parameters, and inference settings.

# 🔹 Paths for storing raw data, processed data, model checkpoints, and outputs
artifacts:
  raw_data_path: "./data/raw/"                         # Directory for raw dataset files
  processed_data_path: "./data/processed/"             # Directory for preprocessed dataset
  files_path: "./artifacts/files/"                     # General storage for generated files
  train_models: "./artifacts/checkpoints/train_models/"  # Directory to store trained models
  best_model: "./artifacts/checkpoints/best_model/"      # Directory for best model checkpoints
  metrics_path: "./artifacts/metrics/"                 # Path to store training/testing metrics
  train_images: "./artifacts/outputs/train_images/"    # Folder to store images generated during training
  test_image: "./artifacts/outputs/test_image/"        # Folder to store predicted test images

# 🔹 Dataset and dataloader settings
dataloader:
  image_path: "./data/raw/dataset.zip"  # Path to the dataset (ZIP format or unzipped folder)
  image_channels: 3                     # Number of image channels (3 for RGB, 1 for grayscale)
  image_size: 128                        # Image resolution (e.g., 128x128)
  batch_size: 8                          # Number of images per batch
  split_size: 0.30                       # Percentage of data used for validation (e.g., 30% validation)

# 🔹 TransUNet Model Configuration
TransUNet:
  nheads: 4              # Number of attention heads in the transformer encoder
  num_layers: 4          # Number of transformer encoder layers
  dim_feedforward: 512   # Hidden layer size in the feedforward network
  dropout: 0.3           # Dropout rate for regularization (higher value prevents overfitting)
  activation: "gelu"     # Activation function ("gelu" or "relu")
  layer_norm_eps: 1e-05  # Epsilon value for layer normalization (stabilizes training)
  bias: False            # Whether to use bias in transformer layers (True/False)

# 🔹 Training Configuration
trainer:
  epochs: 100            # Number of epochs for training
  lr: 0.0001             # Learning rate for optimization
  optimizer: "AdamW"     # Selected optimizer: "Adam", "AdamW", or "SGD"

  # Optimizer configurations (fine-tuning parameters)
  optimizers:
    Adam: 
      beta1: 0.9
      beta2: 0.999
      weight_decay: 0.0001
    SGD: 
      momentum: 0.95
      weight_decay: 0.0
    AdamW:
      beta1: 0.9
      beta2: 0.999
      weight_decay: 0.0001

  # Loss function settings
  loss: 
    type: "bce"           # Type of loss function: "bce", "focal", or "tversky"
    loss_smooth: 1e-06    # Smoothing factor for loss computation (prevents overconfidence)
    alpha_focal: 0.75     # Alpha value for focal loss (balances class distribution)
    gamma_focal: 2        # Gamma value for focal loss (higher values focus on hard examples)
    alpha_tversky: 0.75   # Alpha parameter for Tversky loss (controls false positives)
    beta_tversky: 0.5     # Beta parameter for Tversky loss (controls false negatives)

  # Regularization settings (helps prevent overfitting)
  l1_regularization: False       # Enable L1 regularization (True/False)
  elastic_net_regularization: False  # Enable elastic net regularization (True/False)

  verbose: True       # Display progress and save images during training (True/False)
  device: "cuda"      # Device for training: "cuda" (GPU), "mps" (Mac M1/M2), or "cpu"

# 🔹 Testing Configuration
tester:
  dataset: "test"  # Dataset used for testing
  device: "cuda"   # Device to use for testing: "cuda", "mps", or "cpu"

# 🔹 Inference Configuration
inference:
  image: "./artifacts/data/processed/sample.jpg"  # Path to the image used for inference
```

---

### **📌 Explanation of Key Sections**
| **Section**       | **Description** |
|-------------------|----------------|
| **`artifacts`**  | Defines storage paths for datasets, model checkpoints, and outputs. |
| **`dataloader`** | Specifies dataset path, image properties, batch size, and validation split. |
| **`TransUNet`**  | Defines model architecture, including Transformer layers and activation functions. |
| **`trainer`**    | Configures training parameters like epochs, optimizer, and loss functions. |
| **`tester`**     | Specifies dataset and device settings for evaluation. |
| **`inference`**  | Defines the path to an image for making predictions. |
---

## **📌 Running Training & Testing**
| **Process**      | **Command**                     | **Description** |
|------------------|---------------------------------|-----------------|
| **Train Model**  | `python src/cli.py --train`     | Starts model training using `config.yaml` |
| **Test Model**   | `python src/cli.py --test`      | Runs evaluation on test data |
| **Change Optimizer** | Edit `config.yaml`          | Supports `"SGD"`, `"Adam"`, `"AdamW"` |

---

## **📌 Viewing Results**
### **🔹 Model Checkpoints**
Trained models are saved in:
```bash
./artifacts/checkpoints/train_models/
```

### **🔹 Best Model**
The best-performing model is saved in:
```bash
./artifacts/checkpoints/best_model/
```

### **🔹 Test Predictions**
Generated segmentation masks are saved in:
```bash
./artifacts/outputs/test_image/
```

---

## **📌 TransUNet Workflow**
```
1️⃣ Load Dataset  ----->  2️⃣ Train Model  ----->  3️⃣ Evaluate  ----->  4️⃣ Generate Predictions
```

---

## **📌 Citation**
If you use this repository, please consider citing the **original TransUNet paper**:
```
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yutong and Yu, Qihang and others},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

---

## **📌 License**
This project is **open-source** and available under the **MIT License**.  
🚨 **Note:** This is an **unofficial implementation** and is **not affiliated** with the original authors.

---

# **🔥 Ready to Get Started?**
- Modify `config.yaml` for your setup.  
- Run **training** with: `python src/cli.py --train`  
- Run **testing** with: `python src/cli.py --test`  

🚀 **Happy Training & Testing with TransUNet!** 🚀  