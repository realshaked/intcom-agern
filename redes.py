import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np


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


# 2. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO (seções 7.2 e 8)

def treinar_e_avaliar_rede_neural(cromossomo, X_train, y_train, X_val, y_val,
                                  num_classes=2, epochs=50,
                                  X_teste=None, y_teste=None):
    """
    Treina a rede com os atributos ativos do cromossomo e retorna o F1-Score.

    - Durante o AG: retorna o F1 no conjunto de VALIDAÇÃO (guia da busca).
    - Avaliação final: se (X_teste, y_teste) forem passados, retorna o F1 no
      conjunto de TESTE — dados nunca vistos nem pelo treino nem pela busca
      do AG, dando a estimativa imparcial exigida pela divisão 70/15/15.

    Conforme o enunciado, "a melhor configuração da rede será a que der menor
    erro no conjunto de validação": guardamos os pesos da época com menor
    perda de validação e avaliamos com eles (early stopping por snapshot).
    """
    # 1. Selecionar os atributos indicados pelos genes ativos
    indices_selecionados = np.where(np.asarray(cromossomo) == 1)[0]
    num_atributos_selecionados = len(indices_selecionados)

    # Trava de segurança: cromossomo zerado não gera rede válida
    if num_atributos_selecionados == 0:
        return 0.0

    X_train_filtrado = X_train[:, indices_selecionados]
    X_val_filtrado = X_val[:, indices_selecionados]

    # 2. Configurar hardware (GPU se estiver no Colab, CPU se local)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_filtrado, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_filtrado, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    # 3. Construir a rede neural correspondente
    modelo = RedeNeuralCervical(num_atributos_selecionados, num_classes).to(device)

    # Backpropagation embutido na perda + otimizador Adam com lr = 0,001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)

    # 4. Treinar guardando os pesos da época com menor erro de validação
    melhor_perda_val = float('inf')
    melhores_pesos = None

    for epoch in range(epochs):
        modelo.train()
        optimizer.zero_grad()
        saidas = modelo(X_train_t)
        perda = criterion(saidas, y_train_t)
        perda.backward()
        optimizer.step()

        modelo.eval()
        with torch.no_grad():
            perda_val = criterion(modelo(X_val_t), y_val_t).item()
        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())

    if melhores_pesos is not None:
        modelo.load_state_dict(melhores_pesos)

    # 5. Avaliação: teste (se fornecido) ou validação (durante a busca do AG)
    if X_teste is not None and y_teste is not None:
        X_aval = torch.tensor(X_teste[:, indices_selecionados], dtype=torch.float32).to(device)
        y_aval = torch.tensor(y_teste, dtype=torch.long).to(device)
    else:
        X_aval, y_aval = X_val_t, y_val_t

    modelo.eval()
    with torch.no_grad():
        _, classes_preditas = torch.max(modelo(X_aval), 1)

    # 'macro' trata as classes com peso igual — adequado a bases desbalanceadas
    return f1_score(y_aval.cpu().numpy(), classes_preditas.cpu().numpy(), average='macro')
