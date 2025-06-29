## Handwritten_digit_recognition_using_CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.  
The model is trained using TensorFlow and visualized with Matplotlib.

## Project Overview

- Load and preprocess the MNIST dataset
- Build a CNN model from scratch
- Train and evaluate the model
- Plot accuracy and loss curves

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0–9),  
each of size 28×28 pixels. It is loaded directly using TensorFlow's built-in dataset loader.

## Model Architecture

- Input Layer: 28×28×1 images
- Convolutional Layer (32 filters, 3×3 kernel)
- ReLU Activation
- Pooling Layer
- Fully Connected Layers
- Softmax Output Layer

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow / Keras

## Evaluation

Model performance is visualized using:
- Accuracy vs. Epochs
- Loss vs. Epochs

## Future Enhancements

- Introduce dropout layers to reduce overfitting
- Try advanced optimizers (AdamW, RMSprop)
- Experiment with deeper architectures

## Author

**Sarthak Tiwari** [github_Sorthak]  
If this notebook helped you learn CNNs, give it a star!
