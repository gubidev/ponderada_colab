import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Calcula o erro entre a predição e o valor real.
    y_pred: probabilidades geradas pelo softmax (batch, 10)
    y_true: labels corretos como inteiros (batch,)
    """
    n = y_pred.shape[0]
    # Clipa para evitar log(0) que resultaria em -infinito
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
    # Pega só a probabilidade da classe correta para cada exemplo
    correct_probs = y_pred_clipped[np.arange(n), y_true]
    return -np.mean(np.log(correct_probs))
