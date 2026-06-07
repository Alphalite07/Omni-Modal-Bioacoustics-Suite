# 🧬 Omni-Modal Bioacoustics Suite: Canine Vocalization Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enterprise-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade, locally-hosted analytical dashboard designed for deep acoustic forensics and the automated classification of canine vocalizations. Built with Python, PyTorch, and Streamlit, this suite merges traditional signal processing with deep convolutional neural networks.

---

## 🚀 Overview

The Bioacoustics Suite allows researchers and engineers to upload raw audio (`.wav`) or spectrogram images (`.jpg`/`.png`) to perform instant, high-fidelity classification. 

To combat the limitations of small datasets (Mode Collapse/Few-Shot learning), this software utilizes a **Dual AI Architecture**:
1. **Neural Predictor (Softmax):** A ResNet-18 deep learning model trained via PyTorch to classify spectrogram tensors into distinct vocal classes (Barks, Growls, Howls, Whines).
2. **Structural Matcher (MSE):** Bypasses softmax hallucination by extracting feature arrays from the neural network's penultimate layer and computing Mean Squared Error (MSE) distances against a verified baseline matrix.

---

## ✨ Key Features

* **🎙️ Omni-Modal Slicer:** Automatically processes `.wav` files, applies stationary noise reduction (`noisereduce`), and isolates active vocalization intervals based on customizable decibel thresholds.
* **🔬 Deep Acoustic Forensics:** Generates high-resolution visualizations of raw waveforms, Mel-Spectrograms, and Mel-Frequency Cepstral Coefficients (MFCC) using `librosa`.
* **📊 Explainable AI (XAI):** Renders Plotly-powered distribution charts to visualize the neural network's confidence spread across all potential classes.
* **📂 Automated Batch Processor:** Point the system to a local directory to autonomously analyze hundreds of audio files, export the prediction data to CSV, and visualize dataset distribution.
* **⚙️ Integrated Compiler:** Features a pre-flight boot screen allowing users to compile image matrices and run high-intensity GPU training loops directly from the UI.

---

## 🛠️ Prerequisites & Accessories Needed

To run this suite at full capacity, your system needs the following infrastructure:

### 1. Hardware
* **CPU:** Multi-core processor (Intel i5/AMD Ryzen 5 or better).
* **GPU (Highly Recommended):** NVIDIA GPU with CUDA support for accelerated PyTorch training and tensor inference. (The system will gracefully fall back to CPU if no GPU is detected).

### 2. Software
* **Python:** Version 3.8 or higher.
* **FFmpeg:** Required by `librosa` and `soundfile` to decode complex audio streams. 
    * *Windows:* Install via `winget install ffmpeg`
    * *Linux/WSL:* Install via `sudo apt install ffmpeg`

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Omni-Modal-Bioacoustics-Suite.git](https://github.com/YOUR_USERNAME/Omni-Modal-Bioacoustics-Suite.git)
   cd Omni-Modal-Bioacoustics-Suite


2. **Create and activate a virtual environment:**
   
   *Linux/WSL/macOS:*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   *Windows:*
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```  
3. **Install the Core Dependencies:**
   ```bash
   pip install torch torchvision torchaudio 
   pip install streamlit librosa soundfile noisereduce pandas plotly

   ```



---

## 🏗️ Directory Structure & Matrix Prep

Before launching, you must populate the baseline matrix so the Structural Matcher and Neural Compiler have data to process. Create the following folder structure in the root directory and place your `.wav` and `.jpg`/`.png` files inside their respective class folders:

```text
Omni-Modal-Bioacoustics-Suite/
│
├── raw_images/          # Used for baseline structural matching and AI training
│   ├── barks/           # Add target images here
│   ├── growls/          
│   ├── howls/           
│   └── whines/          
│
├── raw_audio/           # (Optional) Converts to structural baseline matrices
│
├── app.py               # The Monolithic Dashboard
└── README.md

```

---

## 💻 Usage Instructions

1. **Boot the Engine:**
```bash
streamlit run app.py

```


2. **The Initialization Screen:** Upon the very first launch, select "Option B: Train New System". The software will automatically compile the `raw_images` folder into a data matrix, run a 12-epoch ResNet-18 optimization loop, and save the resulting `.pth` weights to your machine.
3. **The Main Suite:** Once booted, navigate between the Omni-Modal Slicer, Acoustic Forensics, and Batch Processor tabs to analyze your target files.

---

## 🛡️ Architecture & Training Notes

To prevent neural network mode collapse on highly limited or single-shot datasets, the internal compiler utilizes aggressive data augmentation, including Random Affine transformations (scaling/shifting) and Color Jittering. The inference pipeline forces Grayscale normalization on all user uploads to maintain strict structural parity with the baseline matrices.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

*Disclaimer: This tool is for research and educational purposes. Do not rely solely on neural predictions for critical veterinary or biological diagnostics.*

