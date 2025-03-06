# **🚀 TransUNet - Transformer-Based U-Net for Medical Image Segmentation**

**TransUNet** is an advanced **hybrid deep learning model** that combines **CNNs (Convolutional Neural Networks)** and **Transformers** to improve segmentation accuracy in medical imaging. The model enhances **U-Net** by integrating **self-attention mechanisms**, making it more efficient for complex segmentation tasks.

---

## **📌 Key Features**
✅ Hybrid **CNN + Transformer** architecture for robust segmentation  
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

### **🔹 Example Configuration: Training Settings**
```yaml
trainer:
  epochs: 100
  lr: 0.0001
  optimizer: "AdamW"
  device: "cuda"  # Change to "cpu" for CPU training
```

### **🔹 Example Configuration: Changing Optimizer**
```yaml
optimizer: "SGD"
```

### **🔹 Example Configuration: Dataset Paths**
```yaml
dataloader:
  image_path: "./data/raw/dataset.zip"
  batch_size: 8
  image_size: 128
  split_size: 0.30
```

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
If you use this repository, please consider citing:
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

---

# **🔥 Ready to Get Started?**
- Modify `config.yaml` for your setup.  
- Run **training** with: `python src/cli.py --train`  
- Run **testing** with: `python src/cli.py --test`  

🚀 **Happy Training & Testing with TransUNet!** 🚀  