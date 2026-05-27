# Project: Melanoma Detection using Deep Learning with PyTorch
# File: model_training.py
# Author: Bernadette Burks
# Created on: May 22, 2026

# Import os to set environment variable for handling duplicate library issues
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import torch modules for building and training the CNN model
import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_preprocessing import preprocess_data


# Model building and training code (6 steps)
total_steps = 6

# Step 1 - Define a simple CNN architecture for melanoma detection
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    # Define the forward pass of the CNN
    def forward(self, x):
        print(x.shape)  # Print input shape for debugging
        x = self.pool(self.relu(self.conv1(x)))
        print(x.shape)  # Print shape after first convolution and pooling
        x = self.pool(self.relu(self.conv2(x)))
        print(x.shape)  # Print shape after second convolution and pooling
        x = x.view(x.size(0), -1)
        print(x.shape)  # Print shape after flattening
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
print(f"Step 1/{total_steps}: CNN model architecture successfully defined.")

# Define function to train the CNN model
def train_model(train_loader, test_loader, num_epochs=10):
    # Initialize the CNN model
    model = SimpleCNN()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    print(f"Step 2/{total_steps}: Model training setup completed. Starting training process...")

    # Set model to train on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Training loop 
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        print(f"Step 3/{total_steps}: Starting epoch {epoch + 1}/{num_epochs}...")

        # Iterate over the training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate loss
            print(f'Batch loss: {loss.item():.4f}', end='\r')  # Print batch loss

        # Print the average loss for the epoch
        print(f"Step 4/{total_steps}: Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print(f"Step 5/{total_steps}: Finished Training")

    # Save the trained model as a .pth file
    torch.save(model.state_dict(), 'models/melanoma_cnn_model.pth')
    print(f"Step 6/{total_steps}: Model saved as models/melanoma_cnn_model.pth")

    # Return the trained model
    return model

if __name__ == "__main__":
    train_loader, test_loader, _, _ = preprocess_data()  # Ensure data is preprocessed before training
    model = train_model(train_loader, test_loader)  # Train the model   