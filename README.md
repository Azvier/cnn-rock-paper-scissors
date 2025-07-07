# Rock, Paper, Scissors - Hand Gesture Classification using CNN

This project showcases the development of a Convolutional Neural Network (CNN) to classify hand gestures for the classic game of Rock, Paper, Scissors. The entire workflow, from data preparation to model training and final prediction, is implemented using Python, TensorFlow, and Keras.

## Features

- **Multiple Model Architectures:** Explores and compares various CNN architectures, including a custom-built CNN and several pre-trained models (VGG16, ResNet50, MobileNetV2, EfficientNetB0, DenseNet121, InceptionV3, and Xception) using transfer learning.
- **Data Augmentation:** Utilizes image augmentation to create a more robust and generalized model.
- **Interactive Prediction:** Provides a script to predict hand gestures from either an uploaded image or a live webcam feed.
- **Detailed Evaluation:** Includes comprehensive model evaluation with performance metrics and confusion matrices.

## Project Workflow

The project follows these key stages:

1.  **Dataset Preparation:** The initial dataset is reorganized into `train`, `validation`, and `test` sets using the `scripts/prepare-dataset.py` script.
2.  **Model Training and Experimentation:** The `notebooks/rock-paper-scissors.ipynb` notebook is used for experimenting with different model architectures, training them on the prepared dataset, and evaluating their performance.
3.  **Prediction:** The `scripts/main.py` script loads the best-performing trained models (VGG16 and ResNet50) to classify new images.

## Project Structure

```
├── image-dataset/
│   └── rockpaperscissors/
├── legacy/
│   └── rock-paper-scissor.ipynb # Legacy notebook
├── models/
│   ├── best_models/      # Stores the best trained model weights
│   └── logs/             # Contains training logs and TensorBoard data
├── notebooks/
│   └── rock-paper-scissors.ipynb  # Jupyter Notebook for model development
├── scripts/
│   ├── prepare-dataset.py # Script to organize the dataset
│   └── main.py            # Script for prediction
├── .gitignore
├── LICENSE
├── pdm.lock
├── pyproject.toml
└── README.md
```

## Note on Reproducibility

Please note that the `image-dataset`, `models/best_models`, and `models/logs` directories are not currently included in this repository due to their large file sizes. They will be uploaded in a future update once they have been properly organized and optimized.

Without these files, the training and prediction scripts cannot be run directly. The primary purpose of this repository, for now, is to showcase the code and the results of the project.

## Models and Results

This project conducted a comparative study of several CNN architectures to find the most effective model for hand gesture classification.

### Models Compared

-   **Custom CNN:** A baseline CNN model built from scratch.
-   **Transfer Learning Models:**
    -   VGG16
    -   ResNet50
    -   MobileNetV2
    -   EfficientNetB0
    -   DenseNet121
    -   InceptionV3
    -   Xception

### Performance Highlights

The models were evaluated based on their accuracy on the test set. The key findings are:

-   **Best Performing Models:** The **VGG16** and **ResNet50** models, after fine-tuning, achieved the highest accuracy, both reaching **100% accuracy** on the test set. Their strong performance is attributed to their deep architectures and the effectiveness of transfer learning.
-   **Other Notable Models:** **MobileNetV2** and **EfficientNetB0** also performed exceptionally well, with test accuracies of **99.09%**. These models offer a good balance between accuracy and computational efficiency.
-   **Custom CNN:** The custom-built CNN achieved a test accuracy of **34.35%**, highlighting the significant advantage of using pre-trained models for this task.

The detailed performance metrics, training history, and confusion matrices for all models are available in the `notebooks/rock-paper-scissors.ipynb` notebook.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
