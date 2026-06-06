import os
import numpy as np
import matplotlib.pyplot as plt

print(">>> Building fake spectrogram dataset for testing...")

# Create the exact folder structure PyTorch needs
folders = [
    'dataset/train/barks', 'dataset/train/growls', 'dataset/train/whines',
    'dataset/val/barks', 'dataset/val/growls', 'dataset/val/whines'
]

# Generate 20 fake images per folder
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    for i in range(20):
        # Create a borderless image
        fig = plt.figure(figsize=(3, 3), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Draw random colored noise (looks just like a real spectrogram)
        noise = np.random.rand(128, 128)
        ax.imshow(noise, cmap='magma', aspect='auto')
        
        # Save it
        plt.savefig(os.path.join(folder, f"dummy_{i:03d}.png"), format='png')
        plt.close(fig)

print(">>> SUCCESS! Your dataset folders are full. You are cleared to run train_ai.py!")