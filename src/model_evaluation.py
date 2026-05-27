# Project: Melanoma Detection using Deep Learning with PyTorch
# File: model_evaluation.py
# Author: Bernadette Burks
# Created on: May 22, 2026

# Evaluate melanoma_cnn_model.pth on the test dataset

# Import necessary libraries for model evaluation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the data preprocessing function to load the test dataset
from src.data_preprocessing import preprocess_data # Import the data preprocessing function to load the test dataset
from src.model_training import SimpleCNN as MelanomaCNN  # Import the CNN architecture from model_training.py

# Load the trained model
model_path = 'models/melanoma_cnn_model.pth'

# Recreate model architecture
model = MelanomaCNN()

# Load weights into model
model.load_state_dict(torch.load(model_path, weights_only=True))

# Set model to evaluation mode
model.eval()

# Load the test dataset
train_loader, test_loader, train_dataset, test_dataset = preprocess_data()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Initialize variables to track evaluation metrics
total_loss = 0.0
correct_predictions = 0
total_samples = 0

# Set model to evaluate on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Evaluating model on device: {device}")

# Evaluation loop
with torch.no_grad():  # Disable gradient calculation for evaluation
    for images, labels in test_loader:
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        total_loss += loss.item()  # Accumulate loss
        
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total_samples += labels.size(0)  # Update total samples count
        correct_predictions += (predicted == labels).sum().item()  # Update correct predictions count

# Calculate average loss and accuracy
average_loss = total_loss / len(test_loader)
accuracy = correct_predictions / total_samples

print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    print("Model evaluation completed successfully.")

# Save evaluation results to a file
with open('results/evaluation_results.txt', 'w') as f:
    f.write(f'Test Loss: {average_loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')

