# Project: Melanoma Detection using Deep Learning
# File: main.py
# Author: Bernadette Burks
# Created on: May 22, 2026

from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.model_prediction import make_prediction

def main():
    # Step 1: Preprocess the data
    train_loader, test_loader = preprocess_data()

    # Step 2: Train the CNN model
    model = train_model(train_loader, test_loader)

    # Step 3: Evaluate the trained model
    accuracy = evaluate_model(model, test_loader)


if __name__ == "__main__":
    print("Welcome to the Melanoma Detection using Deep Learning project!")
    print("This project aims to develop a deep learning model to detect melanoma from skin images.")
    print("Please ensure you have the necessary libraries installed and the dataset ready for training.")
    print("Let's get started with data preprocessing and model training!")
    main()