import os
import shutil
import random
import glob

print(">>> COMPUTER VISION DATASET COMPILER INITIALIZED", flush=True)

if __name__ == "__main__":
    raw_dir = 'raw_images'
    base_out_dir = 'dataset'
    
    if not os.path.exists(raw_dir):
        print(f"ERROR: Cannot find '{raw_dir}' folder. Please create it and add your JPGs.")
        exit()

    classes = ['barks', 'growls', 'howls', 'whines']
    counters = {'train': 0, 'val': 0}

    print(">>> Scanning local directories for spectrogram images...", flush=True)
    
    for class_name in classes:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        image_files = glob.glob(os.path.join(class_dir, '*.jpg')) + glob.glob(os.path.join(class_dir, '*.png'))
        print(f"Found {len(image_files)} images in '{class_name}'. Routing to matrix...", flush=True)
        
        for i, img_path in enumerate(image_files):
            split = 'train' if random.random() < 0.8 else 'val'
            out_folder = os.path.join(base_out_dir, split, class_name)
            os.makedirs(out_folder, exist_ok=True)
            
            file_ext = os.path.splitext(img_path)[1]
            out_file = os.path.join(out_folder, f"custom_{class_name}_{i:03d}{file_ext}")
            shutil.copy(img_path, out_file)
            counters[split] += 1

    print(f"\n--- Matrix Compilation Complete! (Train: {counters['train']} | Val: {counters['val']}) ---", flush=True)