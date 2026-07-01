import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np


# 1. ARQUITETURA DA REDE NEURAL

class RedeNeuralCervical(nn.Module):
    def __init__(self, num_atributos_selecionados, num_classes):
        super(RedeNeuralCervical, self).__init__()
        
        # Camada de Entrada -> Primeira Camada Oculta
        # 32 neurônios
        self.fc1 = nn.Linear(num_atributos_selecionados, 32)
        # Função de ativação ReLU
        self.relu1 = nn.ReLU() 
        
        # Primeira Oculta -> Segunda Camada Oculta
        # 16 neurônios
        self.fc2 = nn.Linear(32, 16)
        # Função de ativação ReLU
        self.relu2 = nn.ReLU() 
        
        # Segunda Oculta -> Camada de Saída
        # Neurônios iguais ao número de classes 
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # Retorna os "logits" puros. O Softmax será aplicado pelo CrossEntropyLoss.
        x = self.fc3(x) 
        return x


# 2. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO

def treinar_e_avaliar_rede_neural(cromossomo, X_train, y_train, X_val, y_val, num_classes=2, epochs=50):
    """
    Substitui a função stub do Algoritmo Genético.
    Treina a rede [cite: 104] e retorna o F1-Score[cite: 105].
    """
    # 1. Selecionar os atributos indicados pelos genes ativos
    indices_selecionados = np.where(cromossomo == 1)[0]
    num_atributos_selecionados = len(indices_selecionados)
    
    # Trava de segurança: se o AG gerar um cromossomo zerado, devolve fitness 0
    if num_atributos_selecionados == 0:
        return 0.0
        
    # Filtrar as matrizes de dados
    X_train_filtrado = X_train[:, indices_selecionados]
    X_val_filtrado = X_val[:, indices_selecionados]
    
    # 2. Configurar hardware (GPU se estiver no Colab, CPU se local)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Converter dados Numpy para Tensores do PyTorch e enviar para o Device
    X_train_t = torch.tensor(X_train_filtrado, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_filtrado, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # 3. Construir a rede neural correspondente
    modelo = RedeNeuralCervical(num_atributos_selecionados, num_classes).to(device)
    
    # Algoritmo de Backpropagation [cite: 86] embutido na perda
    criterion = nn.CrossEntropyLoss() 
    
    # Otimizador Adam com Taxa de aprendizado 0,001
    optimizer = optim.Adam(modelo.parameters(), lr=0.001) 
    
    # 4. Treinar a rede neural
    modelo.train()
    for epoch in range(epochs):
        optimizer.zero_grad() # Zera gradientes passados
        saidas = modelo(X_train_t) # Forward pass
        perda = criterion(saidas, y_train_t) # Calcula o erro
        perda.backward() # Backpropagation
        optimizer.step() # Atualiza os pesos
        
    # 5. Avaliação (Usando o conjunto de validação para medir o erro
    modelo.eval()
    with torch.no_grad():
        previsoes_val = modelo(X_val_t)
        # Pega o índice da classe com maior probabilidade (argmax)
        _, classes_preditas = torch.max(previsoes_val, 1)
        
    # Traz os tensores de volta para a memória da CPU para calcular a métrica
    y_val_cpu = y_val_t.cpu().numpy()
    pred_cpu = classes_preditas.cpu().numpy()
    
    # Calcular o F1-Score [cite: 105]
    # 'macro' trata as classes com peso igual, bom se a base for desbalanceada.
    f1 = f1_score(y_val_cpu, pred_cpu, average='macro')
    
    return f1