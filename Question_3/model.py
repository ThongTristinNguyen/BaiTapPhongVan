import numpy as np
from utils import triplet_loss

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, anchor, positive, negative, alpha=0.2, learning_rate=0.01):
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        
        pos_dist = np.sum((anchor_output - positive_output) ** 2, axis=1)
        neg_dist = np.sum((anchor_output - negative_output) ** 2, axis=1)
        
        dloss_da = 2 * (anchor_output - positive_output)
        dloss_dn = 2 * (anchor_output - negative_output)
        
        dloss = np.maximum(0, pos_dist - neg_dist + alpha)
        
        grad_pos_dist = dloss[:, np.newaxis] * dloss_da
        grad_neg_dist = -dloss[:, np.newaxis] * dloss_dn
        
        dL_dz2 = np.dot(anchor_output.T, grad_pos_dist - grad_neg_dist) / anchor.shape[0]
        dL_db2 = np.sum(grad_pos_dist - grad_neg_dist, axis=0, keepdims=True) / anchor.shape[0]
        dL_dW2 = np.dot(self.a1.T, (grad_pos_dist - grad_neg_dist) / anchor.shape[0])
        
        dL_da1 = np.dot((grad_pos_dist - grad_neg_dist), self.W2.T)
        dL_dz1 = dL_da1 * self.relu_derivative(self.z1)
        
        dL_dW1 = np.dot(anchor.T, dL_dz1) / anchor.shape[0]
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / anchor.shape[0]
        
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def compute_loss(self, anchor, positive, negative, alpha=0.2):
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output, alpha)
        return loss


    def save_weights(self, file_path):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('W1', data=self.W1)
            f.create_dataset('b1', data=self.b1)
            f.create_dataset('W2', data=self.W2)
            f.create_dataset('b2', data=self.b2)
        print(f'Model weights saved to {file_path}')
