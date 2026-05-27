# Project: Melanoma Detection using Deep Learning with PyTorch
# File: model_prediction.py
# Author: Bernadette Burks
# Created on: May 22, 2026

# Make predictions using the trained CNN model on new images of skin lesions

# Import necessary libraries for model prediction
import torch
from torchvision import transforms
from PIL import Image
import os

# Import modules from model_training.py to load the trained model and define the CNN architecture
from src.model_training import SimpleCNN as MelanomaCNN  # Import the CNN architecture from model_training.py

# Import the CNN architecture from model_training.py
model = MelanomaCNN()
image_path = 'skin_lesion.jpg'  # Replace with the actual path to the new image you want to predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU

def predict_image(model, image_path, device):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations (same as during training)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to the same size used during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the model and image to the specified device
    model.to(device)
    image = image.to(device)
    
    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()  # Return the predicted class index

if __name__ == "__main__":
    # Example usage
    model_path = 'melanoma_cnn_model.pth'
    image_path = 'skin_lesion.jpg'
    
    # Make a prediction
    predicted_class = predict_image(model, image_path, device)
    class_labels = {0: 'Benign', 1: 'Malignant'}  
    print(f'Predicted class index for {image_path}: {predicted_class}')
    print(f'Predicted class label for {image_path}: {class_labels[predicted_class]}')

    # Save the prediction result to a file
    with open(f'prediction_result_{os.path.basename(image_path)}.txt', 'w') as f:
        f.write(f'Predicted class index for {image_path}: {predicted_class}\n')
        f.write(f'Predicted class label for {image_path}: {class_labels[predicted_class]}\n')

    