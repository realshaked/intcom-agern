import numpy as np
import matplotlib.pyplot as plt
from utils import pre_processar_dados, separar_conjuntos
from ag import AlgoritmoGenetico

# Configurações Iniciais
CAMINHO_ARQUIVO = 'Base Slim Morte cancer de útero.xlsx' # Mude para o nome exato do arquivo no diretório
NUM_EXPERIMENTOS = 20 # Mude para 20 quando for rodar no Colab valendo

def main():
    # 1. Pré-processamento
    X, y, nomes_atributos, num_classes = pre_processar_dados(CAMINHO_ARQUIVO, target_col='label_cid')
    
    # 2. Divisão dos dados
    X_train, y_train, X_val, y_val, X_test, y_test = separar_conjuntos(X, y)
    
    num_atributos_total = X.shape[1]
    
    historico_experimentos = []
    melhor_cromossomo_geral = None
    melhor_fitness_geral = -1

    print(f"\nIniciando bateria de {NUM_EXPERIMENTOS} experimento(s)...")
    
    for exp in range(NUM_EXPERIMENTOS):
        print(f"\n{'='*20} EXPERIMENTO {exp+1}/{NUM_EXPERIMENTOS} {'='*20}")
        
        # Instancia o Algoritmo Genético
        ag = AlgoritmoGenetico(num_atributos=num_atributos_total)
        
        # Executa a busca
        melhor_cromossomo, melhor_fitness, historico = ag.executar(
            X_train, y_train, X_val, y_val, num_classes
        )
        
        historico_experimentos.append(historico)
        
        if melhor_fitness > melhor_fitness_geral:
            melhor_fitness_geral = melhor_fitness
            melhor_cromossomo_geral = melhor_cromossomo

    # 3. Resultados Finais [cite: 108]
    print("\n" + "="*40)
    print("RESULTADOS FINAIS")
    print("="*40)
    print(f"Melhor Fitness Encontrado: {melhor_fitness_geral:.4f}")
    
    # Identificar quais atributos foram selecionados [cite: 109, 110]
    atributos_selecionados = [nomes_atributos[i] for i in range(num_atributos_total) if melhor_cromossomo_geral[i] == 1]
    print(f"Número de atributos selecionados: {len(atributos_selecionados)}")
    print(f"Atributos utilizados: {atributos_selecionados}")
    
    # 4. Gerar Gráfico de Convergência (Média dos experimentos) 
    tamanho_maximo = max(len(h) for h in historico_experimentos)
    
    # Padroniza o tamanho das listas de histórico (pois alguns experimentos podem parar mais cedo)
    historico_padronizado = []
    for h in historico_experimentos:
        h_ext = list(h) + [h[-1]] * (tamanho_maximo - len(h))
        historico_padronizado.append(h_ext)
        
    curva_media = np.mean(historico_padronizado, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(curva_media, label='Fitness Médio da Elite', color='blue', linewidth=2)
    plt.title('Curva de Convergência do Algoritmo Genético', fontsize=14)
    plt.xlabel('Gerações', fontsize=12)
    plt.ylabel('Fitness Médio', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Salva o gráfico na pasta para você colocar no relatório técnico [cite: 115]
    plt.savefig('convergencia_ag.png', dpi=300, bbox_inches='tight')
    print("\nGráfico de convergência salvo como 'convergencia_ag.png'!")
    plt.show()

if __name__ == "__main__":
    main()