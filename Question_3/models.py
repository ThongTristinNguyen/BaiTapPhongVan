# models.py
import numpy as np
import h5py
from utils import triplet_loss

class CustomNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, anchor, positive, negative, alpha=0.2):
        anchor_output = self.forward_pass(anchor)
        positive_output = self.forward_pass(positive)
        negative_output = self.forward_pass(negative)
        
        loss = triplet_loss(anchor_output, positive_output, negative_output, alpha)
        return loss
    
    def backward(self, anchor, positive, negative, alpha=0.2, learning_rate=0.01):
        anchor_output = self.forward_pass(anchor)
        positive_output = self.forward_pass(positive)
        negative_output = self.forward_pass(negative)
        
        pos_dist = 2 * (anchor_output - positive_output)
        neg_dist = 2 * (anchor_output - negative_output)
        
        dloss_da = pos_dist - neg_dist
        dloss_dp = -pos_dist
        dloss_dn = neg_dist
        
        # weights and bias updates
        self.W2 -= learning_rate * np.dot(self.a1.T, dloss_da)
        self.b2 -= learning_rate * np.sum(dloss_da, axis=0, keepdims=True)
        
        dW1_a = np.dot(anchor.T, np.dot(dloss_da, self.W2.T) * (self.z1 > 0))
        dW1_p = np.dot(positive.T, np.dot(dloss_dp, self.W2.T) * (self.z1 > 0))
        dW1_n = np.dot(negative.T, np.dot(dloss_dn, self.W2.T) * (self.z1 > 0))
        
        db1_a = np.sum(np.dot(dloss_da, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)            
        db1_p = np.sum(np.dot(dloss_dp, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)     
        db1_n = np.sum(np.dot(dloss_dn, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)
        
        self.W1 -= learning_rate * (dW1_a + dW1_p + dW1_n)
        self.b1 -= learning_rate * (db1_a + db1_p + db1_n)
    
    def save_weights(self, filepath):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('W1', data=self.W1)
            f.create_dataset('b1', data=self.b1)
            f.create_dataset('W2', data=self.W2)
            f.create_dataset('b2', data=self.b2)
    
    def load_weights(self, filepath):
        with h5py.File(filepath, 'r') as f:
            self.W1 = f['W1'][:]
            self.b1 = f['b1'][:]
            self.W2 = f['W2'][:]
            self.b2 = f['b2'][:]

def compute_accuracy(anchor, positive, negative, model):
    anchor_output = model.forward_pass(anchor)
    positive_output = model.forward_pass(positive)
    negative_output = model.forward_pass(negative)
    
    pos_dist = np.sum(np.square(anchor_output - positive_output), axis=1)
    neg_dist = np.sum(np.square(anchor_output - negative_output), axis=1)
    
    correct = np.sum(pos_dist < neg_dist)
    accuracy = correct / len(pos_dist)
    
    return accuracy
