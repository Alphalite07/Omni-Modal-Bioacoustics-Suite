import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy

if __name__ == '__main__':
    # 🚨 THE FIX: Aggressive augmentation so it stops memorizing exact pixels!
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Shifts and stretches the waves
            transforms.ColorJitter(brightness=0.3, contrast=0.3), # Prevents color memorization
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'dataset'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Missing matrix target directory '{data_dir}'. Run compiler first.")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Matrix Structural Verification: Success. Target classes detected: {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Infrastructure Mode Locked: Running optimization on context -> [{device.type.upper()}]")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 🚨 THE FIX: Added Weight Decay (L2 Regularization) to prevent mode collapse
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) 

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 12 # Bumped up slightly to allow the new augmentations to settle

    print("Beginning matrix optimization backpropagation sequence...")
    for epoch in range(num_epochs):
        print(f'Epoch Progress Monitor: {epoch+1}/{num_epochs}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            if dataset_sizes[phase] == 0: continue

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
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
            print(f'{phase.capitalize()} Processing -> Objective Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'dog_bioacoustics_model.pth')
    print("Success! Weights structural sync complete. Export matrix generated: dog_bioacoustics_model.pth")