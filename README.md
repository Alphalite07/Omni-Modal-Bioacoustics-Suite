# 🧬 Omni-Modal Bioacoustics Suite: Canine Vocalization & Behavioral Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enterprise-EE4C2C?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/YOLOv11-Custom_Pose-00FFFF?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade, locally-hosted analytical dashboard designed for deep acoustic forensics and temporal multi-modal behavioral tracking. Built with Python, PyTorch, Custom YOLOv11 Pose estimation, and Streamlit, this suite merges digital signal processing with spatial-temporal neural networks to map telemetry directly to established canine ethograms.


---

## 🚀 Overview

The Bioacoustics Suite allows researchers and engineers to upload raw audio (`.wav`), video (`.mp4`), or spectrogram images (`.jpg`/`.png`) to perform instant, high-fidelity classification and movement tracking. 

To combat the limitations of small datasets (Mode Collapse/Few-Shot learning) and handle complex media, this software utilizes a **Tri-Core AI Architecture**:
1. **Acoustic Neural Predictor (ResNet-18 + MSE):** A deep learning model trained via PyTorch to classify spectrogram tensors into distinct vocal classes (Barks, Growls, Howls, Whines). It bypasses softmax hallucination by extracting 512-dimensional feature arrays and computing Mean Squared Error (MSE) distances against a verified baseline matrix.
2. **Kinematic Vision Engine (YOLOv11 Pose):** Bypasses standard human COCO datasets by utilizing a proprietary, custom-trained YOLOv11 Animal Pose model. Extracts exact 17-point quadruped skeletal matrices (X,Y coordinates) and tracks trajectories dynamically across video frames.
3. **Temporal Behavioral Brain (LSTM):** A Long Short-Term Memory (LSTM) recurrent neural network that ingests 1-second chunks of skeletal matrix data (`.npy`) to understand the dimension of time. It maps the kinetic dance of the skeleton directly to baseline behaviors from the **IISER Kolkata Dog Lab Canine Ethogram**.
---

## ✨ Key Features

* **🎥 Temporal Ethogram Mapping:** Automatically extracts spatial coordinates from videos, translates them into 34-dimensional temporal arrays, and uses PyTorch LSTM inference to predict ethogram states (e.g., Stand, Walk, Sit, Roll).
* **🎙️ Omni-Modal Slicer:** Automatically ingests .wav or .mp4 files, uses moviepy to strip audio tracks, applies stationary noise reduction (noisereduce), and isolates active vocalization intervals based on customizable decibel thresholds.
* **🦴 Kinematic Vision Tracking:** Dedicated high-speed YOLOv11 Pose dashboard for extracting quadruped skeletal movement vectors in video files using CUDA acceleration.
* **🌌 Dimensional Visualization:** Deploys `scikit-learn` Principal Component Analysis (PCA) to compress neural tensors into an interactive 3D spatial galaxy using `Plotly`.
* **🔬 Deep Acoustic Forensics:** Generates high-resolution visualizations of raw waveforms, Mel-Spectrograms, and Mel-Frequency Cepstral Coefficients (MFCC) using `librosa`.
* **📡 Live Acoustic Radar:** Real-time WebRTC audio interceptor for live environmental acoustic telemetry.
* **📂 Automated Batch Processor:** Point the system to a local directory to autonomously analyze hundreds of audio files and export the prediction data to CSV.
* **⚡ Enterprise Memory Management:** Implements `@st.cache_resource` and manual memory-flush UI controls to prevent RAM overloads and VRAM thrashing during heavy matrix multiplications.

---

## 🛠️ Prerequisites & Accessories Needed

To run this suite at full capacity, your system needs the following infrastructure:

### 1. Hardware
* **CPU:** Multi-core processor (Intel i5/AMD Ryzen 5 or better).
* **GPU (Highly Recommended):** NVIDIA GPU with dedicated VRAM and CUDA support for accelerated PyTorch training and YOLOv11 inference. (The system will gracefully fall back to CPU if no GPU is detected).

### 2. Software
* **Python:** Version 3.8 or higher.
* **FFmpeg:** Required by `librosa`, `soundfile`, and `moviepy` to decode complex media streams. 
    * *Windows:* Install via `winget install ffmpeg`
    * *Linux/WSL:* Install via `sudo apt install ffmpeg`

---

## 📦 Installation & Setup

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/Alphalite07/Omni-Modal-Bioacoustics-Suite.git](https://github.com/Alphalite07/Omni-Modal-Bioacoustics-Suite.git)
cd Omni-Modal-Bioacoustics-Suite

```

**2. Create and activate a virtual environment:**
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

**3. Install the Core Dependencies:**
*(Note: `moviepy` is strictly pinned to v1.0.3 to ensure stable audio extraction).*

```bash
pip install streamlit librosa soundfile noisereduce pandas scikit-learn plotly ultralytics opencv-python-headless streamlit-webrtc av moviepy==1.0.3

```

**4. Install CUDA-Accelerated PyTorch:**
*If running on a machine with a dedicated Nvidia GPU, ensure you install the CUDA-specific version of PyTorch for optimal inference speeds:*

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

---

## 🏗️ Deployment Architecture & Matrix Prep

Before launching, ensure your proprietary neural weights are placed directly in the root directory:
* `custom_canine_pose.pt` (The Spatial YOLOv11 Engine)
* `canine_behavior_v1.pth` (The Temporal LSTM Engine)

You must also populate the baseline matrix so the Structural Matcher and Neural Compiler have data to process. Create the following folder structure in the root directory:

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

## ⚠️ WSL Memory Configuration Note

If running this suite within the **Windows Subsystem for Linux (WSL)**, the intense video extraction and neural matrix caching will cause massive RAM spikes, leading to system freezes. It is **highly recommended** to configure a `.wslconfig` file in your Windows `%userprofile%` directory to forcefully cap physical memory consumption:

```ini
[wsl2]
memory=8GB
pageReporting=true

```

---

## 💻 Usage Instructions

**1. Boot the Engine:**

```bash
streamlit run app.py

```

**2. The Initialization Screen:** Upon the very first launch, select **"Option B: Train New System"**. The software will automatically compile the `raw_images` folder into a data matrix, run a 12-epoch ResNet-18 optimization loop, and save the resulting `.pth` weights to your machine.

**3. The Main Suite:** Once booted, navigate between the Omni-Modal Slicer, Acoustic Forensics, Live Radar, and Vision tabs to execute analytical scans.

---

## 🛡️ Architecture & Training Notes

To prevent neural network mode collapse on highly limited or single-shot datasets, the internal compiler utilizes aggressive data augmentation, including Random Affine transformations (scaling/shifting) and Color Jittering. The inference pipeline forces Grayscale normalization on all user uploads to maintain strict structural parity with the baseline matrices.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

*Disclaimer: This tool is for research and educational purposes. Do not rely solely on neural predictions for critical veterinary or biological diagnostics.*

