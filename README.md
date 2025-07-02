# Rock, Paper, Scissors - Hand Gesture Classification using CNN

This project involves designing, training, and evaluating a Convolutional Neural Network (CNN) to classify images of hand gestures into three categories: Rock, Paper, or Scissors. It serves as a fundamental example of image classification using deep learning.

## Technical Overview

-   **Model Architecture:** Designed and trained a Sequential CNN model using TensorFlow and the Keras API in a Google Colab environment, leveraging GPU acceleration for efficient training.
-   **Image Augmentation:** Applied image augmentation techniques during training. This artificially expands the training dataset by creating modified versions of images (e.g., rotating, shearing, zooming), which helps the model generalize better and improves its robustness against variations in new images.
-   **Prediction Script:** Developed a final prediction script that loads the trained model weights and accurately classifies new, unseen hand gesture images.

## Technologies & Libraries Used

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   Matplotlib
-   Google Colab

## Project Workflow

Image Dataset → Data Augmentation → CNN Model Training → Model Evaluation → Prediction on New Images

---

***Disclaimer:** Please note that the code currently in this repository is the raw version from an earlier stage of my learning journey. A more well-documented, refactored, and structured version is currently in development.*
