import tensorflow as tf
import numpy as np

def load_data():
    """
    Load the MNIST dataset using TensorFlow 2.x APIs.
    
    Returns:
        mnist (Dataset): MNIST dataset object containing train, validation, and test sets.
    """
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    return (train_data, train_labels), (test_data, test_labels)

def preprocess_data(train_data, test_data):
    """
    Preprocess the MNIST dataset.
    
    Args:
        train_data (numpy.ndarray): Training data.
        test_data (numpy.ndarray): Test data.
    
    Returns:
        train_data (numpy.ndarray): Preprocessed training data.
        test_data (numpy.ndarray): Preprocessed test data.
    """
    # Normalize pixel values to the range [0, 1]
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Reshape data to add a channel dimension (required for CNN)
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)

    return train_data, test_data

if __name__ == '__main__':
    # Load the MNIST dataset
    (train_data, train_labels), (test_data, test_labels) = load_data()

    # Preprocess the data
    train_data, test_data = preprocess_data(train_data, test_data)

    # Print dataset details
    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)