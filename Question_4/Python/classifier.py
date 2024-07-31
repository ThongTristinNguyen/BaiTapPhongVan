import numpy as np

class NaiveBayesClassifier:
    def __init__(self, alpha=1e-5):
        self.alpha = alpha  

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        
        num_samples, num_features = X_train.shape
        self.classes, counts = np.unique(y_train, return_counts=True)
        num_classes = len(self.classes)
        
        self.class_probs = counts / num_samples
        self.feature_probs = np.zeros((num_classes, num_features, 256))  

        for c in range(num_classes):
            X_c = X_train[y_train == self.classes[c]]
            for i in range(num_features):
                for value in range(256):
                    self.feature_probs[c, i, value] = (np.sum(X_c[:, i] == value) + self.alpha) / (X_c.shape[0] + 256 * self.alpha)

    def predict(self, X_test):
        X_test = np.array(X_test)
        num_samples, num_features = X_test.shape
        num_classes = len(self.classes)
        
        log_probs = np.zeros((num_samples, num_classes))
        
        for c in range(num_classes):
            class_prob = np.log(self.class_probs[c])
            feature_probs = np.sum(np.log(self.feature_probs[c, np.arange(num_features), X_test]), axis=1)
            log_probs[:, c] = class_prob + feature_probs
        
        return self.classes[np.argmax(log_probs, axis=1)]
