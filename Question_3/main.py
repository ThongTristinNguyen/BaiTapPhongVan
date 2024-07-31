import numpy as np
import time
import os
from utils import read_mnist_images, read_mnist_labels, preprocess_data
from models import CustomNN, compute_accuracy

start_time = time.time()

# Load data
train_images_path = r'/content/drive/MyDrive/BaiTapPhongVan/MNIST/train-images.idx3-ubyte'
train_labels_path = r'/content/drive/MyDrive/BaiTapPhongVan/MNIST/train-labels.idx1-ubyte'
test_images_path = r'/content/drive/MyDrive/BaiTapPhongVan/MNIST/t10k-images.idx3-ubyte'
test_labels_path = r'/content/drive/MyDrive/BaiTapPhongVan/MNIST/t10k-labels.idx1-ubyte'

train_images = read_mnist_images(train_images_path)
train_labels = read_mnist_labels(train_labels_path)
test_images = read_mnist_images(test_images_path)
test_labels = read_mnist_labels(test_labels_path)

# Preprocess data
X_train = preprocess_data(train_images)
X_test = preprocess_data(test_images)

# Initialize model
input_size = 784
hidden_size = 128
output_size = 64
model = CustomNN(input_size, hidden_size, output_size)

# Shuffle training data
permutation = np.random.permutation(X_train.shape[0])
X_train = X_train[permutation]
train_labels = train_labels[permutation]

batch_size = 64
num_epochs = 20

# Training
for epoch in range(num_epochs):
    learning_rate = 0.001
    LOSS = []
    for i in range(0, X_train.shape[0], batch_size):
        end = i + batch_size
        if end > X_train.shape[0]:
            break
        
        anchor_batch = X_train[i:end]
        positive_batch = X_train[i:end]
        negative_batch = X_train[(i + batch_size) % X_train.shape[0]: (i + 2 * batch_size) % X_train.shape[0]]
        
        if len(negative_batch) < len(anchor_batch):
            continue
        
        loss = model.compute_loss(anchor_batch, positive_batch, negative_batch)
        LOSS.append(loss)
        model.backward(anchor_batch, positive_batch, negative_batch, learning_rate=learning_rate)
        
    print(f'Epoch {epoch + 1}:')
    print(f'Loss: {np.mean(LOSS):.4f}')

# Save model weights
save_path = r'/content/drive/MyDrive/BaiTapPhongVan/Weight/weight.h5'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_weights(save_path)
print('Model weights saved.')

# Evaluate on test data
anchor_test = X_test[:batch_size]
positive_test = X_test[:batch_size]
negative_test = X_test[batch_size:2 * batch_size]

test_loss = model.compute_loss(anchor_test, positive_test, negative_test)
test_accuracy = compute_accuracy(anchor_test, positive_test, negative_test, model)

print(f'Accuracy: {test_accuracy * 100:.2f}%')
