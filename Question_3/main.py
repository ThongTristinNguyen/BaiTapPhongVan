import os
from model import SimpleNN
from utils import load_data, triplet_loss

def evaluate_accuracy(model, X, y):
    correct = 0
    total = 0
    for i in range(0, X.shape[0], batch_size):
        end = i + batch_size
        if end > X.shape[0]:
            break

        anchor_batch = X[i:end]
        positive_batch = X[i:end]
        negative_batch = X[(i + batch_size) % X.shape[0]: (i + 2 * batch_size) % X.shape[0]]

        # Get predictions for the test batch
        anchor_output = model.forward(anchor_batch)
        positive_output = model.forward(positive_batch)
        negative_output = model.forward(negative_batch)
        
        # Implement a dummy accuracy calculation or use a more suitable metric for your task
        correct += np.sum(np.argmax(anchor_output, axis=1) == np.argmax(positive_output, axis=1))
        total += len(anchor_batch)

    accuracy = correct / total
    return accuracy

def train(model, X_train, y_train, X_test, y_test, num_epochs, batch_size, learning_rate):
    for epoch in range(num_epochs):
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
            
            loss, _ = model.compute_loss(anchor_batch, positive_batch, negative_batch)
            LOSS.append(loss)
            model.backward(anchor_batch, positive_batch, negative_batch, learning_rate=learning_rate)
        
        train_accuracy = evaluate_accuracy(model, X_train, y_train)
        test_accuracy = evaluate_accuracy(model, X_test, y_test)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(LOSS)}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
    
    # Save the model weights after training
    model.save_weights('weights/nn_model.h5')

def main():
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_data()
    
    input_size = 28 * 28
    hidden_size = 128
    output_size = 64
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 32
    
    model = SimpleNN(input_size, hidden_size, output_size)
    
    # Create directory for saving weights if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    # Train the model
    train(model, X_train, y_train, X_test, y_test, num_epochs, batch_size, learning_rate)

if __name__ == "__main__":
    main()
