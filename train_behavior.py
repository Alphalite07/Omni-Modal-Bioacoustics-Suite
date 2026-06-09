import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os

# --- 1. THE NEURAL ARCHITECTURE ---
class CanineBehaviorLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=4):
        super(CanineBehaviorLSTM, self).__init__()
        # Input size is 34 because we have 17 joints * 2 coordinates (X, Y)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass sequence through the LSTM
        out, (hn, cn) = self.lstm(x)
        # We only care about the network's conclusion at the VERY LAST frame of the sequence
        out = self.fc(out[:, -1, :]) 
        return out

# --- 2. DATA LOAD & PREP ---
def load_and_slice_data(sequence_length=30):
    print("🧠 Loading Temporal Matrices...")
    X_data = []
    Y_labels = []
    
    # Map file names to Ethogram classes
    class_map = {'stand': 0, 'walk': 1, 'sit': 2, 'roll': 3}
    
    for file_path in glob.glob("behavior_data/*.npy"):
        file_name = os.path.basename(file_path).replace('.npy', '')
        if file_name not in class_map: continue
        
        label = class_map[file_name]
        matrix = np.load(file_path) # Shape: [Total_Frames, 17, 2]
        
        # Flatten the 17x2 coordinates into a single 34-number array per frame
        matrix = matrix.reshape(matrix.shape[0], 34) 
        
        # Slice the long video into 30-frame (1 second) "thought chunks"
        for i in range(0, len(matrix) - sequence_length, sequence_length):
            chunk = matrix[i:i + sequence_length]
            X_data.append(chunk)
            Y_labels.append(label)
            
    return torch.tensor(X_data, dtype=torch.float32), torch.tensor(Y_labels, dtype=torch.long)

# --- 3. THE TRAINING LOOP ---
def train_engine():
    X, Y = load_and_slice_data(sequence_length=30)
    print(f"📊 Dataset Built: {len(X)} sequential thought-chunks harvested.")
    
    # Initialize the brain and send to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanineBehaviorLSTM().to(device)
    X, Y = X.to(device), Y.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 150
    print(f"🚀 Firing up LSTM Engine on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: AI guesses the behavior
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Backward pass: AI learns from its mistakes
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # Save the proprietary brain!
    torch.save(model.state_dict(), 'canine_behavior_v1.pth')
    print("✅ Training Complete! Saved behavioral brain as 'canine_behavior_v1.pth'")

if __name__ == "__main__":
    train_engine()