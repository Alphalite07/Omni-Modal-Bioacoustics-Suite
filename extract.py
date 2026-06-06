import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import os

def apply_highpass_filter(audio_signal, sample_rate, cutoff_freq=150.0):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normalized_cutoff, btype='high', analog=False)
    return filtfilt(b, a, audio_signal)

def extract_vocalizations(audio_path, output_dir='extracted_clips', cutoff=150, silence_thresh=25, min_length=0.3):
    print(f"Parsing streaming metrics for: {audio_path}...")
    
    if not os.path.exists(audio_path):
        print(f"Critical Error: Execution path targeted to '{audio_path}' could not be resolved.")
        return

    y, sr = librosa.load(audio_path, sr=None)
    y_clean = apply_highpass_filter(y, sr, cutoff_freq=cutoff)
    intervals = librosa.effects.split(y_clean, top_db=silence_thresh)
    
    os.makedirs(output_dir, exist_ok=True)
        
    count = 0
    for interval in intervals:
        start_sample, end_sample = interval
        segment = y_clean[start_sample:end_sample]
        duration = len(segment) / sr
        
        if duration >= min_length:
            out_file = os.path.join(output_dir, f"dog_vocalization_{count:03d}.wav")
            sf.write(out_file, segment, sr)
            count += 1
            
    print(f"Execution complete. Extracted and isolated {count} audio blocks inside directory mapping: '{output_dir}'.")

if __name__ == "__main__":
    extract_vocalizations('1.wav')