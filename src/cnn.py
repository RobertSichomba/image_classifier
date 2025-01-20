import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os

def build_cnn_model():
    """
    Build the CNN model architecture using TensorFlow 2.x.
    
    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_cnn():
    """
    Train and evaluate the CNN model on the MNIST dataset.
    """
    # Load the MNIST dataset
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # Preprocess the data
    train_data = train_data / 255.0  # Normalize pixel values to [0, 1]
    test_data = test_data / 255.0
    train_data = train_data.reshape(-1, 28, 28, 1)  # Add channel dimension
    test_data = test_data.reshape(-1, 28, 28, 1)

    # Build and train the model
    model = build_cnn_model()
    model.fit(train_data, train_labels, epochs=10, batch_size=50, validation_split=0.1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Save the model
    model_dir = './src/model'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'mnist_cnn_model.h5'))
    print(f"Model saved to {model_dir}")

if __name__ == '__main__':
    train_and_evaluate_cnn()
