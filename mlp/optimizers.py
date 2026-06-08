import numpy as np


class SGD:
    """
    Stochastic Gradient Descent: atualiza os pesos subtraindo
    o gradiente multiplicado pela taxa de aprendizado.
    """
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def update(self, weights, biases, dW, db):
        for i in range(len(weights)):
            weights[i] -= self.lr * dW[i]
            biases[i] -= self.lr * db[i]


class SGDMomentum:
    """
    SGD com momentum: acumula velocidade na direção do gradiente,
    ajuda a sair de mínimos locais mais rápido.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, dW, db):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.lr * dW[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * db[i]
            weights[i] += self.velocity_w[i]
            biases[i] += self.velocity_b[i]
