import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

print(">>> LOCAL BIOACOUSTICS COMPILER INITIALIZED", flush=True)

def create_spectrogram_image(audio_path, output_path):
    # Load audio and generate mathematical matrix
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Render borderless image
    fig = plt.figure(figsize=(3, 3), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    raw_dir = 'raw_audio'
    base_out_dir = 'dataset'
    
    if not os.path.exists(raw_dir):
        print(f"ERROR: Cannot find '{raw_dir}' folder. Please create it and add your audio files.")
        exit()

    classes = ['barks', 'growls', 'whines']
    counters = {'train': 0, 'val': 0}

    print(">>> Scanning local directories for raw acoustic data...", flush=True)
    
    for class_name in classes:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Grab all .wav files in the folder
        wav_files = glob.glob(os.path.join(class_dir, '*.wav'))
        print(f"Found {len(wav_files)} files in '{class_name}'. Processing...", flush=True)
        
        for i, wav_path in enumerate(wav_files):
            # Mathematically perfect 80/20 split
            split = 'train' if random.random() < 0.8 else 'val'
            
            out_folder = os.path.join(base_out_dir, split, class_name)
            os.makedirs(out_folder, exist_ok=True)
            
            out_file = os.path.join(out_folder, f"custom_{class_name}_{i:03d}.png")
            create_spectrogram_image(wav_path, out_file)
            counters[split] += 1

    print(f"\n--- Matrix Compilation Complete! (Train: {counters['train']} | Val: {counters['val']}) ---", flush=True)