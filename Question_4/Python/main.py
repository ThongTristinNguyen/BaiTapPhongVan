from fastapi import FastAPI, UploadFile, File
from dataloader import MnistDataloader
from classifier import NaiveBayesClassifier
import numpy as np

app = FastAPI()

@app.post("/train")
async def train(training_images_filepath: str, training_labels_filepath: str, test_images_filepath: str, test_labels_filepath: str):
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    x_train_normalized = (x_train / np.max(x_train) * 255).astype(int)
    x_test_normalized = (x_test / np.max(x_test) * 255).astype(int)

    x_train_flat = x_train_normalized.reshape(len(x_train_normalized), -1)
    x_test_flat = x_test_normalized.reshape(len(x_test_normalized), -1)

    nb = NaiveBayesClassifier(alpha=1e-5)
    nb.fit(x_train_flat, y_train)
    
    y_pred = nb.predict(x_test_flat)
    train_accuracy = np.mean(nb.predict(x_train_flat) == y_train)
    test_accuracy = np.mean(y_pred == y_test)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }
