from datasets import load_dataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random

print(">>> DOG BIOACOUSTICS DATA PIPELINE INITIALIZED", flush=True)

def create_spectrogram_image(audio_array, sr, output_path):
    S = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = plt.figure(figsize=(3, 3), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    print(">>> Establishing connection to Barkopedia repositories...", flush=True)
    dataset = load_dataset("ArlingtonCL2/Barkopedia-Dog-Vocal-Detection", split="validation")

    base_dir = 'dataset'
    counters = {'train': 0, 'val': 0}
    total_clips = len(dataset)

    print(f">>> Target sync established! Discovered {total_clips} dog records. Processing visual arrays...", flush=True)
    
    for i, item in enumerate(dataset):
        audio_array = np.array(item['audio']['array'])
        sr = item['audio']['sampling_rate']
        
        # --- SAFE INTEGER MAPPER ---
        # Checks if the label is missing or corrupted (None) before converting
        raw_label = item.get('label')
        label_id = int(raw_label) if raw_label is not None else 0
        
        if label_id == 0:
            label_name = "barks"
        elif label_id == 1:
            label_name = "growls"
        elif label_id == 2:
            label_name = "whines"
        else:
            label_name = ["barks", "growls", "whines"][label_id % 3] 
        # -----------------------------
        
        split = 'train' if random.random() < 0.8 else 'val'
        folder_path = os.path.join(base_dir, split, label_name)
        os.makedirs(folder_path, exist_ok=True)
        
        output_path = os.path.join(folder_path, f"hf_clip_{i:05d}.png")
        create_spectrogram_image(audio_array, sr, output_path)
        counters[split] += 1
        
        if i % 25 == 0 and i > 0:
            print(f">>> Processing stream row: {i} / {total_clips} completed. (Train: {counters['train']} | Val: {counters['val']})", flush=True)

    print("\n--- Structural Extraction Pipeline Complete! ---", flush=True)