# Project: Melanoma Detection using Deep Learning with PyTorch
# File: data_preprocessing.py
# Author: Bernadette Burks
# Created on: May 22, 2026

# Import torch modules for building the CNN model
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Insert loading bar for data preprocessing steps below
def loading_bar(progress, total, bar_length=30):
    percent = float(progress) / total
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'Loading and preprocessing data: [{arrow}{spaces}] {int(percent * 100)}%', end='\r')

# Function to load and preprocess the dataset (4 steps)
def preprocess_data():
    total_steps = 4

    # Step 1 - Define transformations for the training and testing datasets
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Resize images to 32x32 pixels
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize pixel values
    ])

    loading_bar(1, total_steps)

    # Step 2 - Load the dataset (replace with actual dataset path or URL)
    train_dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\berna\OneDrive - New England College\Desktop\Project Root Folders\Melanoma\data\train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\berna\OneDrive - New England College\Desktop\Project Root Folders\Melanoma\data\test', transform=transform)

    loading_bar(2, total_steps)

    # Step 3 - Create data loaders for training and testing datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    loading_bar(3, total_steps)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    loading_bar(4, total_steps)
    print()  # Move to the next line after the loading bar

    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    train_loader, test_loader, train_dataset, test_dataset = preprocess_data()
    
    # Check class labels and dataset sizes
    print("Data preprocessing completed successfully. Class labels:", train_dataset.classes, test_dataset.classes)
    print("Number of training samples:", len(train_loader.dataset))
    print("Number of testing samples:", len(test_loader.dataset))
    print("Ready to proceed with model training.")