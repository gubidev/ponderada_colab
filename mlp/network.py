import numpy as np
from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy_loss


class MLP:
    """
    Multi-Layer Perceptron implementado manualmente com NumPy.

    Parâmetros:
        layer_sizes: lista com tamanho de cada camada, ex: [784, 128, 64, 10]
        learning_rate: taxa de aprendizado do SGD
    """

    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # Inicialização He: escala os pesos para funcionar bem com ReLU.
        # Pesos zerados fazem todos os neurônios aprenderem a mesma coisa.
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Passagem para frente: calcula a saída da rede para a entrada X.
        Guarda os valores intermediários (ativações e z's) para o backprop.
        """
        self.activations = [X]  # a[0] = entrada
        self.zs = []            # z[i] = pré-ativação da camada i

        current = X
        # Camadas ocultas usam ReLU
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.zs.append(z)
            current = relu(z)
            self.activations.append(current)

        # Camada de saída usa Softmax (converte em probabilidades)
        z = current @ self.weights[-1] + self.biases[-1]
        self.zs.append(z)
        output = softmax(z)
        self.activations.append(output)

        return output

    def backward(self, y_true):
        """
        Backpropagation: calcula os gradientes e atualiza os pesos.
        O gradiente flui da saída para a entrada (de trás para frente).
        """
        n = y_true.shape[0]
        num_layers = len(self.weights)
        dW = [None] * num_layers
        db = [None] * num_layers

        # Gradiente na saída: derivada de (softmax + cross-entropy) = y_pred - y_real
        # Isso é uma simplificação elegante que vem da combinação das duas funções.
        y_one_hot = np.zeros_like(self.activations[-1])
        y_one_hot[np.arange(n), y_true] = 1
        delta = (self.activations[-1] - y_one_hot) / n

        dW[-1] = self.activations[-2].T @ delta
        db[-1] = np.sum(delta, axis=0, keepdims=True)

        # Propaga o gradiente para as camadas ocultas (de trás para frente)
        for i in range(num_layers - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * relu_derivative(self.zs[i])
            dW[i] = self.activations[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

        # Atualiza pesos com SGD
        for i in range(num_layers):
            self.weights[i] -= self.lr * dW[i]
            self.biases[i] -= self.lr * db[i]

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
        """
        Treina a rede por mini-batches e retorna o histórico de loss e acurácia.
        """
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        n = X_train.shape[0]

        for epoch in range(epochs):
            # Embaralha os dados a cada época para o SGD funcionar bem
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Processa em mini-batches
            for start in range(0, n, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                y_pred = self.forward(X_batch)
                epoch_loss += cross_entropy_loss(y_pred, y_batch)
                num_batches += 1

                self.backward(y_batch)

            avg_loss = epoch_loss / num_batches
            train_acc = self.evaluate(X_train, y_train)
            val_acc = self.evaluate(X_val, y_val)

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Época {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Treino: {train_acc:.4f} | Teste: {val_acc:.4f}")

        return history

    def predict(self, X):
        """Retorna a classe predita (índice com maior probabilidade)."""
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        """Retorna a acurácia (fração de acertos)."""
        return np.mean(self.predict(X) == y)
