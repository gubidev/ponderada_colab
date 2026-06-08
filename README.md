# MLP do Zero — MNIST

Implementação de uma rede neural MLP (Multi-Layer Perceptron) sem frameworks de deep learning. Apenas NumPy para os cálculos.

## Como rodar

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar o notebook de experimentos
jupyter notebook notebooks/experimentos.ipynb
```

O notebook carrega o MNIST automaticamente via `keras.datasets.mnist`.

## Estrutura do projeto

```
.
├── README.md
├── mlp/
│   ├── __init__.py
│   ├── network.py        ← MLP com forward e backpropagation
│   ├── activations.py    ← ReLU, derivada do ReLU, Softmax
│   ├── losses.py         ← Cross-entropy loss
│   └── optimizers.py     ← SGD e SGD com Momentum
├── notebooks/
│   └── experimentos.ipynb
├── results/
│   └── (plots gerados durante o experimento)
└── requirements.txt
```

## Arquitetura escolhida

A rede tem a seguinte estrutura de camadas:

```
Entrada (784) → Camada Oculta 1 (128, ReLU) → Camada Oculta 2 (64, ReLU) → Saída (10, Softmax)
```

**Por que essas escolhas:**

- **784 entradas:** cada imagem MNIST é 28×28 pixels = 784 valores
- **ReLU nas camadas ocultas:** simples e eficiente. Não sofre do problema de gradiente que desaparece (vanishing gradient) como a sigmoid
- **Softmax na saída:** converte a saída bruta em probabilidades que somam 1, ideal para classificação com múltiplas classes
- **Cross-entropy loss:** métrica padrão para classificação — penaliza mais quando a confiança na classe errada é alta
- **Inicialização He:** os pesos começam com escala `sqrt(2/n_entradas)`, ideal para ReLU. Pesos zerados fazem todos os neurônios aprenderem a mesma coisa (problema de simetria)

## Resultados

| Configuração | Camadas | Learning Rate | Acurácia no Teste |
|---|---|---|---|
| Experimento 1 | [784, 128, 64, 10] | 0.1 | ~97% |
| Experimento 2 | [784, 256, 128, 10] | 0.05 | ~97.5% |

As curvas de loss e acurácia estão em `results/`.

## Decisões e dificuldades

### 1. Qual foi a decisão técnica mais difícil?

A parte mais difícil foi implementar o backpropagation corretamente, especialmente entender o que é o `delta` (gradiente do erro) em cada camada e como ele flui de trás para frente.

A decisão mais importante foi sobre a inicialização dos pesos. Tentei inicializar todos com zero e a rede simplesmente não aprendeu nada — a loss ficava constante. Pesquisando, entendi o problema: se todos os pesos são iguais, todos os neurônios calculam exatamente a mesma coisa e recebem o mesmo gradiente. A rede inteira se comporta como se tivesse um único neurônio. A solução foi usar a inicialização He: pesos aleatórios com variância controlada (`sqrt(2/n)`), que funciona bem especialmente com ReLU.

### 2. O que não funcionou? O que aprendi?

Primeiro tentei usar uma taxa de aprendizado muito pequena (0.001) e a rede mal saía de ~70% após 20 épocas. Depois testei 0.5 e a loss ficava oscilando sem convergir. Aprendi que o learning rate é um dos hiperparâmetros mais sensíveis: muito pequeno = aprende devagar demais, muito grande = nunca converge. O valor 0.1 funcionou bem para a configuração com batch de 64.

Também tive dificuldade com a estabilidade numérica do softmax. Quando os valores de entrada eram muito grandes, `np.exp()` retornava `inf`. A solução foi subtrair o valor máximo antes de calcular a exponencial — matematicamente equivalente, mas numericamente estável.

### 3. Se fosse refazer do zero, o que faria diferente?

Começaria com um problema menor, como o XOR (4 exemplos, 1 camada oculta), antes de ir direto para o MNIST. É muito mais fácil debugar quando você sabe exatamente qual deve ser a saída. Também implementaria um gradient check desde o início — comparar o gradiente analítico com a aproximação numérica `(f(x+ε) - f(x-ε)) / 2ε` teria me poupado bastante tempo tentando entender por que a rede não aprendia.
