import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_to_pure_spectrogram(audio_path, output_image_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = plt.figure(figsize=(3, 3), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
    plt.savefig(output_image_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    input_folder = 'raw_barks'
    output_folder = 'ai_images'

    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(input_folder):
        print(f"Scanning local input mapping structure '{input_folder}' for raw records...")
        for filename in os.listdir(input_folder):
            if filename.endswith(".wav"):
                audio_path = os.path.join(input_folder, filename)
                image_filename = filename.replace('.wav', '.png')
                image_path = os.path.join(output_folder, image_filename)
                
                print(f"Processing structural conversion: {filename} -> {image_filename}")
                convert_to_pure_spectrogram(audio_path, image_path)
                
        print("\nLocal Processing Sequence Finalized. Sort image sheets manually to train/val storage maps.")
    else:
        print(f"System Error: Path route '{input_folder}' not resolved. Create structure and load raw files.")