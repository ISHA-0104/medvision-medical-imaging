# 🧠 Medical Image Enhancement using Deep Learning

A deep learning-based medical image enhancement system designed to improve the visual quality of MRI images. The application provides an intuitive Streamlit interface where users can upload MRI scans and obtain enhanced images for better visualization.

---

## 📌 Overview

Medical images often suffer from low contrast, noise, and intensity inhomogeneity, making diagnosis challenging. This project applies a deep learning model to enhance MRI images while preserving important anatomical structures.

The project consists of:

- Deep Learning enhancement model
- Streamlit-based web application
- Automated inference pipeline
- Image comparison visualization
- Model training and evaluation pipeline

---

## ✨ Features

- Upload MRI images through a user-friendly Streamlit interface
- Automatic preprocessing
- Deep learning-based enhancement
- Side-by-side comparison of input and enhanced image
- Fast inference
- Save enhanced outputs
- Support for NumPy-based image storage

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Deep Learning
- PyTorch

### Image Processing
- NumPy
- OpenCV
- Pillow

### Utilities
- Matplotlib
- JSON Configuration

---

## 📂 Project Structure

```
MedicalEnhancement/
│
├── dataset/                 # Dataset
├── models/                  # Model architecture
├── outputs/
│   ├── enhanced_images/     # Enhanced MRI images
│   └── inference_comparison.png
│
├── logs/                    # Training logs
├── plots/                   # Training plots
├── utils/                   # Utility functions
├── config.json              # Configuration file
├── app.py                   # Streamlit application
├── inference.py             # Inference pipeline
├── train.py                 # Model training
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/MedicalEnhancement.git

cd MedicalEnhancement
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🚀 Workflow

1. Upload an MRI image.
2. The image is preprocessed.
3. The trained deep learning model performs enhancement.
4. The enhanced image is generated.
5. Input and enhanced images are displayed side-by-side.
6. Enhanced output is saved in the `outputs/` directory.

---

## 📊 Model Output

Generated outputs include:

- Enhanced MRI image (.npy)
- Comparison visualization (.png)
- Training plots
- Logs

---

## 📈 Future Improvements

- Support for 3D MRI volumes
- Multiple enhancement models
- Real-time inference
- DICOM support
- Quantitative evaluation metrics (PSNR, SSIM)
- GPU deployment

---

 👨‍💻 Team
Pooja Sheet
Isha B Kamath
Anusha G S
Srujan Bhat
Developed as part of a Medical Image Enhancement Hackathon project.

---

## 📄 License

This project is intended for educational and research purposes.
