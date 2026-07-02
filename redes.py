import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# 1. ARQUITETURA DA REDE NEURAL (seção 7.1 do enunciado)

class RedeNeuralCervical(nn.Module):
    def __init__(self, num_atributos_selecionados, num_classes):
        super(RedeNeuralCervical, self).__init__()

        # Camada de Entrada -> Primeira Camada Oculta: 32 neurônios + ReLU
        self.fc1 = nn.Linear(num_atributos_selecionados, 32)
        self.relu1 = nn.ReLU()

        # Primeira Oculta -> Segunda Camada Oculta: 16 neurônios + ReLU
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()

        # Segunda Oculta -> Camada de Saída: neurônios = número de classes
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # Retorna os "logits" puros. O Softmax exigido pelo enunciado é aplicado
        # implicitamente pelo CrossEntropyLoss (equivalente a Softmax + log-loss);
        # aplicá-lo duas vezes degradaria o treinamento.
        x = self.fc3(x)
        return x


# 2. TREINAMENTO (helper interno reutilizado pela avaliação e pelo diagnóstico)

def _treinar_modelo(cromossomo, X_train, y_train, X_val, y_val, num_classes,
                    epochs=150, batch_size=64, paciencia=15):
    """
    Treina a rede com os atributos ativos do cromossomo e devolve
    (modelo, device, indices_selecionados). Retorna (None, ...) se o
    cromossomo não selecionar nenhum atributo.

    Decisões de treinamento (correção do classificador degenerado):
    - Mini-batches: cada época tem vários passos do otimizador (e não apenas um,
      como no treino em lote completo), permitindo a convergência efetiva.
    - Perda ponderada por classe (pesos = inverso da frequência): impede que a
      rede minimize a perda prevendo sempre a classe majoritária.
    - Snapshot da melhor época por perda de validação, com early stopping por
      paciência: atende ao critério do enunciado ("melhor configuração = menor
      erro na validação") e evita desperdício de épocas.
    """
    indices_selecionados = np.where(np.asarray(cromossomo) == 1)[0]
    if len(indices_selecionados) == 0:
        return None, None, indices_selecionados

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train[:, indices_selecionados], dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val[:, indices_selecionados], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )

    modelo = RedeNeuralCervical(len(indices_selecionados), num_classes).to(device)

    # Pesos por classe = inverso da frequência; classes ausentes no treino -> peso 0
    contagem = torch.bincount(y_train_t, minlength=num_classes).float()
    pesos = torch.where(contagem > 0, len(y_train_t) / (num_classes * contagem),
                        torch.zeros_like(contagem)).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos)
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)

    melhor_perda_val = float('inf')
    melhores_pesos = None
    epocas_sem_melhora = 0

    for epoch in range(epochs):
        modelo.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            perda = criterion(modelo(xb), yb)
            perda.backward()
            optimizer.step()

        modelo.eval()
        with torch.no_grad():
            perda_val = criterion(modelo(X_val_t), y_val_t).item()

        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                break  # early stopping

    if melhores_pesos is not None:
        modelo.load_state_dict(melhores_pesos)
    return modelo, device, indices_selecionados


def _prever(modelo, device, indices, X):
    X_t = torch.tensor(X[:, indices], dtype=torch.float32).to(device)
    modelo.eval()
    with torch.no_grad():
        _, classes = torch.max(modelo(X_t), 1)
    return classes.cpu().numpy()


# 3. FUNÇÃO DE AVALIAÇÃO USADA PELO AG (seções 7.2 e 8)

def treinar_e_avaliar_rede_neural(cromossomo, X_train, y_train, X_val, y_val,
                                  num_classes=2, X_teste=None, y_teste=None):
    """
    Treina a rede e retorna o F1-Score (macro).

    - Durante o AG: retorna o F1 no conjunto de VALIDAÇÃO (guia da busca).
    - Avaliação final: se (X_teste, y_teste) forem passados, retorna o F1 no
      conjunto de TESTE — dados nunca vistos nem pelo treino nem pela busca.
    """
    modelo, device, indices = _treinar_modelo(
        cromossomo, X_train, y_train, X_val, y_val, num_classes
    )
    if modelo is None:  # cromossomo zerado
        return 0.0

    if X_teste is not None and y_teste is not None:
        y_true, y_pred = y_teste, _prever(modelo, device, indices, X_teste)
    else:
        y_true, y_pred = y_val, _prever(modelo, device, indices, X_val)

    # 'macro' trata as classes com peso igual — adequado a bases desbalanceadas
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def diagnostico_teste(cromossomo, X_train, y_train, X_val, y_val,
                      X_teste, y_teste, num_classes, nomes_classes=None):
    """
    Treina a rede com o cromossomo e imprime, no conjunto de teste, a matriz de
    confusão e o F1 por classe. Serve para verificar se a rede realmente
    discrimina as classes (e não colapsa na majoritária). Retorna o dicionário
    do classification_report.
    """
    modelo, device, indices = _treinar_modelo(
        cromossomo, X_train, y_train, X_val, y_val, num_classes
    )
    y_pred = _prever(modelo, device, indices, X_teste)
    alvos = list(range(num_classes))
    print("\nMatriz de confusão (teste) — linhas = verdadeiro, colunas = predito:")
    print(confusion_matrix(y_teste, y_pred, labels=alvos))
    print("\nRelatório por classe (teste):")
    print(classification_report(y_teste, y_pred, labels=alvos,
                                target_names=nomes_classes, zero_division=0))
    return classification_report(y_teste, y_pred, labels=alvos,
                                 target_names=nomes_classes, zero_division=0,
                                 output_dict=True)
