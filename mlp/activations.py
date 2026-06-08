import numpy as np


def relu(z):
    """Zera valores negativos, mantém positivos."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivada do ReLU: 1 onde z > 0, 0 caso contrário."""
    return (z > 0).astype(float)


def softmax(z):
    """
    Transforma a saída bruta em probabilidades que somam 1.
    Subtrai o máximo por estabilidade numérica (evita overflow).
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
