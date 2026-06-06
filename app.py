import os
import matplotlib
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import noisereduce as nr
import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import Image
import glob
import time
import shutil
import random
import copy

# ==========================================
# 0. SYSTEM CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title="Bioacoustics Enterprise Suite", page_icon="🧬", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['barks', 'growls', 'howls', 'whines'] 
MODEL_PATH = 'dog_bioacoustics_model.pth'

if 'app_state' not in st.session_state:
    st.session_state.app_state = 'BOOT'

# ==========================================
# 1. COMPILER & TRAINING ENGINE
# ==========================================
def compile_dataset(raw_dir='raw_images', base_out_dir='dataset', status_text=None, progress_bar=None):
    if not os.path.exists(raw_dir):
        return False, f"Missing {raw_dir} directory."
    if os.path.exists(base_out_dir):
        shutil.rmtree(base_out_dir)
        
    counters = {'train': 0, 'val': 0}
    total_files = sum([len(files) for r, d, files in os.walk(raw_dir)])
    processed = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_dir): continue
        
        # Robust file grabbing
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            
        for i, img_path in enumerate(image_files):
            split = 'train' if random.random() < 0.8 else 'val'
            out_folder = os.path.join(base_out_dir, split, class_name)
            os.makedirs(out_folder, exist_ok=True)
            
            file_ext = os.path.splitext(img_path)[1]
            out_file = os.path.join(out_folder, f"custom_{class_name}_{i:03d}{file_ext}")
            shutil.copy(img_path, out_file)
            counters[split] += 1
            processed += 1
            if status_text:
                status_text.text(f"Compiling Matrix: {processed}/{total_files} images routed...")
                progress_bar.progress(processed / max(1, total_files))
                
    return True, f"Matrix Built (Train: {counters['train']} | Val: {counters['val']})"

def train_pytorch_model(status_text, progress_bar, metrics_box):
    data_dir = 'dataset'
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    best_model_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0
    num_epochs = 12

    for epoch in range(num_epochs):
        status_text.text(f"Neural Optimization: Epoch {epoch+1}/{num_epochs} running...")
        progress_bar.progress((epoch) / num_epochs)
        epoch_str = f"**Epoch {epoch+1}/{num_epochs}**\n"
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0
            if dataset_sizes[phase] == 0: continue

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_str += f"- {phase.upper()} | Loss: `{epoch_loss:.4f}` | Acc: `{epoch_acc:.4f}`\n"

            if phase == 'val' and epoch_acc >= best_acc:
                best_acc, best_model_wts = epoch_acc, copy.deepcopy(model.state_dict())
        metrics_box.markdown(epoch_str)

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_PATH)
    if device.type == 'cuda': torch.cuda.empty_cache()

# ==========================================
# 2. DUAL AI ENGINE (Predictor + Matcher)
# ==========================================
@st.cache_resource
def load_ai_engines(model_path='dog_bioacoustics_model.pth'):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model = model.to(device)
    model.eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    return model, extractor

def prepare_tensor(pil_image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image.convert('RGB')).unsqueeze(0).to(device)

def run_dual_analysis(model, extractor, pil_image, pool):
    tensor = prepare_tensor(pil_image)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        prob_dict = {CLASS_NAMES[i]: probs[i].item() * 100 for i in range(len(CLASS_NAMES))}
        pred_conf, pred_idx = torch.max(probs, 0)
        pred_label = CLASS_NAMES[pred_idx.item()]
        pred_conf = pred_conf.item() * 100
        features = extractor(tensor).cpu().numpy().flatten()
        
    best_match_label, best_match_name, min_dist, best_match_img = "UNKNOWN", "None", float('inf'), None
    
    for label, references in pool.items():
        for ref in references:
            dist = np.mean((features - ref["features"]) ** 2)
            if dist < min_dist:
                min_dist, best_match_label, best_match_name, best_match_img = dist, label, ref["name"], ref["image"]
                
    match_conf = max(0, 100 - (min_dist * 1000))
    return pred_label, pred_conf, prob_dict, best_match_label, best_match_name, match_conf, best_match_img

# ==========================================
# 3. AUDIO & BASELINE GENERATORS
# ==========================================
def audio_to_pil_spectrogram(y, sr, cmap='gray'):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(3, 3), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap=cmap)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

@st.cache_data
def load_baseline_pool(_extractor):
    pool = {}
    
    # 🚨 FIX 1: Robust OS Walk to catch ALL images, ignoring case issues
    if os.path.exists('raw_images'):
        for root, dirs, files in os.walk('raw_images'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    lbl = os.path.basename(root).upper()
                    try:
                        # Force RGB conversion to prevent RGBA tensor crashes
                        img = Image.open(img_path).convert('RGB')
                        feats = _extractor(prepare_tensor(img)).cpu().numpy().flatten()
                        if lbl not in pool: pool[lbl] = []
                        pool[lbl].append({"name": f"📸 {file}", "features": feats, "image": img})
                    except Exception as e:
                        print(f"Warning: Failed to load baseline image {file} - {str(e)}")

    if os.path.exists('raw_audio'):
        for root, dirs, files in os.walk('raw_audio'):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    lbl = os.path.basename(root).upper()
                    try:
                        y, sr = librosa.load(wav_path, sr=None)
                        img = audio_to_pil_spectrogram(y, sr, cmap='gray')
                        feats = _extractor(prepare_tensor(img)).cpu().numpy().flatten()
                        if lbl not in pool: pool[lbl] = []
                        pool[lbl].append({"name": f"🎵 {file}", "features": feats, "image": img})
                    except Exception as e:
                        print(f"Warning: Failed to load baseline audio {file} - {str(e)}")
                        
    return pool

def render_acoustic_profile(y, sr, cmap):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#1f77b4')
    axes[0].set_title('Raw Acoustic Waveform (Amplitude vs Time)', weight='bold')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axes[1], cmap=cmap)
    axes[1].set_title('Mel-Spectrogram (Frequency vs Time)', weight='bold')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', ax=axes[2], cmap='coolwarm')
    axes[2].set_title('Mel-Frequency Cepstral Coefficients (MFCC Signature)', weight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf

def process_audio_pro(file_path, silence_thresh, min_length, model, extractor, pool, status_text, progress_bar, ui_cmap):
    status_text.text("Phase 1/4: Decoding audio bitstream...")
    progress_bar.progress(0.15)
    y, sr = librosa.load(file_path, sr=None)
    
    status_text.text("Phase 2/4: Applying stationary noise reduction...")
    progress_bar.progress(0.40)
    y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8, stationary=True)
    
    status_text.text("Phase 3/4: Isolating vocalization intervals...")
    progress_bar.progress(0.60)
    intervals = librosa.effects.split(y_clean, top_db=silence_thresh)
    
    extracted_clips = []
    valid_intervals = []
    metadata = []
    count = 1
    total_intervals = len(intervals)
    
    for idx, interval in enumerate(intervals):
        status_text.text(f"Phase 4/4: Dual AI Analysis (Segment {idx+1}/{total_intervals})...")
        progress_bar.progress(0.60 + (0.40 * (idx / max(1, total_intervals))))
        
        start, end = interval
        segment = y_clean[start:end]
        duration = len(segment) / sr
        
        if duration >= min_length:
            ai_img_obj = audio_to_pil_spectrogram(segment, sr, cmap='gray')
            p_lbl, p_conf, prob_dict, m_lbl, m_name, m_conf, m_img = run_dual_analysis(model, extractor, ai_img_obj, pool)
            
            extracted_clips.append({
                'audio': segment, 'sr': sr, 'duration': duration, 'id': count,
                'p_label': p_lbl, 'p_conf': p_conf, 'prob_dict': prob_dict, 
                'm_label': m_lbl, 'm_conf': m_conf, 'm_name': m_name, 'm_img': m_img,
                'interval': interval # 🚨 FIX 2: Store interval directly to prevent Numpy Array crashing later
            })
            valid_intervals.append((interval, p_lbl))
            
            metadata.append({
                "ID": count, "Start(s)": round(start / sr, 2), "Dur(s)": round(duration, 2),
                "Pred_Class": p_lbl.upper(), "Pred_Conf(%)": round(p_conf, 1),
                "Match_Class": m_lbl.upper(), "Match_Conf(%)": round(m_conf, 1), "Matched_File": m_name
            })
            count += 1
            
    return extracted_clips, y_clean, sr, valid_intervals, pd.DataFrame(metadata)

def create_master_timeline(y, sr, intervals_with_labels, cmap, time_range=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap=cmap)
    
    for interval, label in intervals_with_labels:
        start, end = interval
        t_start, dur = start / sr, (end - start) / sr
        c = '#00FF00' if label == 'barks' else '#FF9900' if label == 'growls' else '#CC00FF' if label == 'howls' else '#00CCFF' 
        ax.add_patch(patches.Rectangle((t_start, 0), dur, 8000, linewidth=2, edgecolor=c, facecolor=c, alpha=0.3))
        ax.text(t_start, 8200, label.upper(), color=c, fontsize=10, weight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
    if time_range: ax.set_xlim(time_range[0], time_range[1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    return buf

# ==========================================
# 4. APP ROUTING: BOOT & TRAINING
# ==========================================
if st.session_state.app_state == 'BOOT':
    st.title("⚙️ Bioacoustics Pro Initialization")
    st.markdown("Welcome to the Enterprise Bioacoustics Engine. Please select a startup sequence.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🟢 Option A: Load Existing System")
        st.markdown("Instantly boots the application using previously generated PyTorch weight maps.")
        if os.path.exists(MODEL_PATH):
            if st.button("Launch Application Suite", type="primary", use_container_width=True):
                st.session_state.app_state = 'MAIN'
                st.rerun()
        else:
            st.error("❌ No neural weights found. You must train the system first.")
            st.button("Launch Application Suite", disabled=True, use_container_width=True)

    with col2:
        st.warning("### ⚠️ Option B: Train New System")
        st.markdown("Compiles the `raw_images` matrix and executes a high-intensity GPU training loop.")
        if st.button("Initialize Neural Training", type="primary", use_container_width=True):
            st.session_state.app_state = 'TRAINING'
            st.rerun()

elif st.session_state.app_state == 'TRAINING':
    st.title("🧠 Neural Network Optimization")
    st.markdown("Executing Core Compilation and Backpropagation. Do not close this window.")
    
    status_text, progress_bar, metrics_box = st.empty(), st.progress(0), st.empty()
    
    status_text.text("Phase 1: Compiling Dataset Matrix...")
    success, msg = compile_dataset(status_text=status_text, progress_bar=progress_bar)
    
    if not success:
        st.error(msg)
        if st.button("Return to Menu"):
            st.session_state.app_state = 'BOOT'
            st.rerun()
    else:
        st.success(msg)
        train_pytorch_model(status_text, progress_bar, metrics_box)
        status_text.text("✅ Optimization Complete! Launching Suite...")
        progress_bar.progress(1.0)
        time.sleep(1)
        st.session_state.app_state = 'MAIN'
        st.rerun()

# ==========================================
# 5. APP ROUTING: MAIN DASHBOARD
# ==========================================
elif st.session_state.app_state == 'MAIN':
    
    model, extractor = load_ai_engines()
    baseline_pool = load_baseline_pool(extractor)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Neural_network.svg/512px-Neural_network.svg.png", width=60)
        st.markdown("### 🚨 Path Debugger")
        st.write("**Current Directory:**", os.getcwd())
        st.write("**Can I see raw_images?**", os.path.exists("raw_images"))
        st.markdown("## Global Settings")
        spectrogram_cmap = st.selectbox("UI Spectrogram Colors", ["magma", "viridis", "inferno", "plasma", "coolwarm", "gray"])
        
        st.divider()
        st.markdown("## System Diagnostics")
        st.metric("CUDA Acceleration", "ONLINE" if device.type == "cuda" else "OFFLINE", "GPU Engaged" if device.type == "cuda" else "- CPU Only")
        st.metric("Baseline Pool Size", sum(len(v) for v in baseline_pool.values()), "Vectors Loaded")
        
        if not baseline_pool: st.error("No baselines detected in raw_images/raw_audio!")
        
        st.markdown("### Profile Matrix")
        for lbl, items in baseline_pool.items():
            st.write(f"**{lbl}**: {len(items)} profiles")
            
        st.divider()
        st.markdown("### Admin Controls")
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as fp:
                st.download_button("📥 Export Neural Weights (.pth)", fp, file_name=MODEL_PATH, mime="application/octet-stream", use_container_width=True)
        if st.button("🔄 Reboot System (Retrain)", use_container_width=True):
            st.session_state.app_state = 'BOOT'
            st.rerun()

    st.title("🧬 Ultimate Bioacoustics Pro Suite")
    st.markdown("Enterprise-grade spectral analysis, neural prediction, and structural matching matrix.")

    tab1, tab2, tab3 = st.tabs(["👁️ Omni-Modal Analyzer & Slicer", "🔬 Deep Acoustic Forensics", "📂 Batch Processor"])

    # --- TAB 1: OMNI-MODAL ANALYZER & SLICER ---
    with tab1:
        if 'processed' not in st.session_state: st.session_state.processed = False
        uploaded_file = st.file_uploader("Upload Audio (.wav) or Spectrogram (.jpg/.png)", type=["wav", "jpg", "jpeg", "png"], key="omni")

        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.markdown("---")
            status_text, progress_bar = st.empty(), st.progress(0)

            if file_type in ['jpg', 'jpeg', 'png']:
                status_text.text("Processing Visual Tensor... 50%")
                progress_bar.progress(0.5)
                time.sleep(0.1)
                
                img = Image.open(uploaded_file).convert('RGB')
                p_lbl, p_conf, prob_dict, m_lbl, m_name, m_conf, m_img = run_dual_analysis(model, extractor, img, baseline_pool)
                
                status_text.text("✅ Analysis Complete! 100%")
                progress_bar.progress(1.0)
                time.sleep(0.3)
                status_text.empty(); progress_bar.empty()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img, caption="Target Spectrogram Tensor", use_container_width=True)
                    if m_img:
                        st.markdown("### 🎯 Structural Match")
                        st.image(m_img, caption=f"Match: {m_name}", use_container_width=True)
                        st.info(f"Distance Score: {m_conf:.2f}% ({m_lbl.upper()})")
                with col2:
                    st.markdown(f"### 🧠 AI Prediction: {p_lbl.upper()} ({p_conf:.1f}%)")
                    df_probs = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Confidence'])
                    fig = px.bar(df_probs, x='Confidence', y='Class', orientation='h', title="Explainable AI: Softmax Confidence Distribution", color='Confidence', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)

            elif file_type == 'wav':
                col_set1, col_set2, col_set3 = st.columns(3)
                with col_set1: silence_thresh = st.slider("Silence Threshold (dB)", 10, 50, 25, key="t1")
                with col_set2: min_length = st.slider("Min Clip Length (sec)", 0.1, 1.0, 0.3, key="t2")
                with col_set3: min_conf = st.slider("Min Match Confidence (%)", 0, 100, 0, key="t3")
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("🚀 Run Complete Audio Extraction", type="primary"):
                    local_temp = "temp_input.wav"
                    with open(local_temp, "wb") as f: f.write(uploaded_file.getbuffer())
                    
                    clips, y_clean, sr, ivals, df = process_audio_pro(local_temp, silence_thresh, min_length, model, extractor, baseline_pool, status_text, progress_bar, spectrogram_cmap)
                    
                    status_text.text("✅ Analysis Complete! 100%")
                    progress_bar.progress(1.0)
                    time.sleep(0.3)
                    status_text.empty(); progress_bar.empty()
                    
                    # 🚨 FIX 3: Clean, bug-free mapping using the stored interval dict!
                    if not df.empty:
                        df = df[df['Match_Conf(%)'] >= min_conf]
                        clips = [c for c in clips if c['m_conf'] >= min_conf]
                        ivals = [(c['interval'], c['p_label']) for c in clips] 
                    
                    st.session_state.update({'clips': clips, 'y_clean': y_clean, 'sr': sr, 'intervals': ivals, 'df': df, 'duration': len(y_clean) / sr, 'processed': True})
                    if os.path.exists(local_temp): os.remove(local_temp)
                    
                if st.session_state.processed:
                    if not st.session_state.clips:
                        st.warning("No localized vocalizations found passing thresholds.")
                    else:
                        st.success(f"Isolated and analyzed {len(st.session_state.clips)} segments.")
                        st.markdown("### Master Timeline Visualization")
                        zoom = st.slider("🔎 Timeline Zoom", 0.0, float(st.session_state.duration), (0.0, float(st.session_state.duration)), 1.0, label_visibility="collapsed")
                        timeline = create_master_timeline(st.session_state.y_clean, st.session_state.sr, st.session_state.intervals, spectrogram_cmap, zoom)
                        st.image(timeline, use_container_width=True)
                        
                        col_t1, col_t2 = st.columns([2, 1])
                        with col_t1:
                            st.markdown("### Dual AI Profile Matrix")
                            st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)
                        with col_t2:
                            st.markdown("### Data Export")
                            st.download_button("📊 Download Database (CSV)", st.session_state.df.to_csv(index=False), "data.csv", "text/csv", use_container_width=True)
                            zip_buf = io.BytesIO()
                            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                                for c in st.session_state.clips:
                                    cb = io.BytesIO()
                                    sf.write(cb, c['audio'], c['sr'], format='WAV')
                                    zf.writestr(f"clip_{c['id']:03d}.wav", cb.getvalue())
                            zip_buf.seek(0)
                            st.download_button("📦 Download Audio Chunks (ZIP)", zip_buf, "audio.zip", "application/zip", type="primary", use_container_width=True)

                        st.divider()
                        st.markdown("### 🔬 Segment Diagnostic Inspector")
                        opts = [f"Segment {c['id']:03d} | Pred: {c['p_label'].upper()} ({c['p_conf']:.1f}%) | Match: {c['m_label'].upper()}" for c in st.session_state.clips]
                        sel_idx = st.selectbox("Inspect Segment:", range(len(opts)), format_func=lambda x: opts[x])
                        sel_clip = st.session_state.clips[sel_idx]
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("#### Audio & Match")
                            pb = io.BytesIO()
                            sf.write(pb, sel_clip['audio'], sel_clip['sr'], format='WAV')
                            st.audio(pb.getvalue(), format="audio/wav")
                            st.write(f"**Duration:** {sel_clip['duration']:.2f}s")
                            if sel_clip['m_img']: 
                                st.image(sel_clip['m_img'], caption=f"Baseline Match: {sel_clip['m_name']} ({sel_clip['m_conf']:.1f}%)", use_container_width=True)
                        with c2:
                            st.markdown(f"#### XAI Distribution (Predicted: {sel_clip['p_label'].upper()})")
                            df_probs = pd.DataFrame(list(sel_clip['prob_dict'].items()), columns=['Class', 'Confidence'])
                            fig = px.bar(df_probs, x='Confidence', y='Class', orientation='h', color='Confidence', color_continuous_scale='viridis', height=300)
                            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: DEEP ACOUSTIC FORENSICS ---
    with tab2:
        st.markdown("### Acoustic Signature Breakdown")
        st.markdown("Extract and visualize the raw physics of the audio signal.")
        
        forensic_file = st.file_uploader("Upload Audio (.wav) for Forensic Analysis", type=["wav"], key="forensic")
        if forensic_file:
            y, sr = librosa.load(forensic_file, sr=None)
            st.audio(forensic_file)
            with st.spinner("Rendering acoustic physics arrays..."):
                profile_img = render_acoustic_profile(y, sr, spectrogram_cmap)
                st.image(profile_img, use_container_width=True)

    # --- TAB 3: BATCH DIRECTORY PROCESSOR ---
    with tab3:
        st.markdown("### Automated Database Builder")
        st.markdown("Point the system to a local folder to run high-speed, autonomous analysis on multiple .WAV files.")
        
        folder_path = st.text_input("Enter absolute folder path (e.g., /mnt/c/Users/.../audio):")
        
        if st.button("Initialize Batch Sequence", type="primary"):
            if os.path.isdir(folder_path):
                wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
                if not wav_files:
                    st.warning("No .wav files found in directory.")
                else:
                    b_prog, b_stat = st.progress(0), st.empty()
                    results = []
                    
                    for idx, path in enumerate(wav_files):
                        b_stat.text(f"Processing: {os.path.basename(path)} ({idx+1}/{len(wav_files)})")
                        b_prog.progress((idx + 1) / len(wav_files))
                        try:
                            y, sr = librosa.load(path, sr=None)
                            img = audio_to_pil_spectrogram(y, sr, cmap='gray') 
                            p_lbl, p_conf, _, m_lbl, m_name, m_conf, _ = run_dual_analysis(model, extractor, img, baseline_pool)
                            results.append({
                                "File": os.path.basename(path), "Prediction": p_lbl.upper(), "Pred_Conf": round(p_conf, 2),
                                "Best_Match": m_name, "Match_Score": round(m_conf, 2)
                            })
                        except Exception as e:
                            st.error(f"Failed processing {os.path.basename(path)}: {e}")
                    
                    b_stat.text("✅ Batch sequence complete!")
                    df_batch = pd.DataFrame(results)
                    st.dataframe(df_batch, use_container_width=True)
                    st.download_button("Download Database (CSV)", df_batch.to_csv(index=False).encode('utf-8'), "batch_analysis.csv", "text/csv", type="primary")
                    
                    st.markdown("### Database Analytics")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.plotly_chart(px.pie(df_batch, names='Prediction', title='Class Prediction Distribution'), use_container_width=True)
                    with col_p2:
                        st.plotly_chart(px.pie(df_batch, names='Match_Class' if 'Match_Class' in df_batch else 'Prediction', title='Structural Match Distribution'), use_container_width=True)
            else:
                st.error("Invalid directory path.")