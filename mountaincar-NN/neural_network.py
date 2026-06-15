import numpy as np

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}
    
    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        
        self.cache = {'x': x, 'z1': z1, 'a1': a1}
        return z2.squeeze(0) if single else z2
    
    def backward(self, dL_dz2):
        single = dL_dz2.ndim == 1
        if single:
            dL_dz2 = dL_dz2.reshape(1, -1)
        
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        batch = x.shape[0]
        
        dL_dW2 = a1.T @ dL_dz2 / batch
        dL_db2 = dL_dz2.mean(axis=0)
        
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (z1 > 0)
        
        dL_dW1 = x.T @ dL_dz1 / batch
        dL_db1 = dL_dz1.mean(axis=0)
        
        # Gradient clipping — prevents exploding updates
        for grad in [dL_dW1, dL_db1, dL_dW2, dL_db2]:
            np.clip(grad, -1.0, 1.0, out=grad)
        
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2


