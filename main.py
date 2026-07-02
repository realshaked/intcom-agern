import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import pre_processar_dados, separar_conjuntos
from ag import AlgoritmoGenetico
from redes import treinar_e_avaliar_rede_neural, diagnostico_teste

# Configurações Iniciais
CAMINHO_ARQUIVO = 'Base Slim Morte cancer de útero.xlsx'
NUM_EXPERIMENTOS = 20  # Enunciado: média dos melhores em 20 experimentos completos


def main():
    # 1. Pré-processamento (5 etapas da seção 3)
    X, y, nomes_atributos, num_classes, nomes_classes = pre_processar_dados(
        CAMINHO_ARQUIVO, target_col='label_cid'
    )

    # 2. Divisão dos dados: 70% treino / 15% validação / 15% teste (seção 8)
    X_train, y_train, X_val, y_val, X_test, y_test = separar_conjuntos(X, y)

    num_atributos_total = X.shape[1]

    # O AG enxerga apenas treino+validação; o teste fica reservado para o fim.
    def avaliador(cromossomo):
        return treinar_e_avaliar_rede_neural(
            cromossomo, X_train, y_train, X_val, y_val, num_classes
        )

    historico_experimentos = []
    melhores_por_experimento = []   # melhor cromossomo de CADA experimento
    fitness_por_experimento = []
    melhor_cromossomo_geral = None
    melhor_fitness_geral = -1

    print(f"\nIniciando bateria de {NUM_EXPERIMENTOS} experimento(s)...")

    for exp in range(NUM_EXPERIMENTOS):
        print(f"\n{'='*20} EXPERIMENTO {exp+1}/{NUM_EXPERIMENTOS} {'='*20}")

        # Cada experimento é uma execução independente do AG (população nova).
        ag = AlgoritmoGenetico(num_atributos=num_atributos_total, funcao_avaliacao=avaliador)
        melhor_cromossomo, melhor_fitness, historico = ag.executar()

        historico_experimentos.append(historico)
        melhores_por_experimento.append(melhor_cromossomo)
        fitness_por_experimento.append(melhor_fitness)

        if melhor_fitness > melhor_fitness_geral:
            melhor_fitness_geral = melhor_fitness
            melhor_cromossomo_geral = melhor_cromossomo

    # 3. Consolidação entre experimentos: com que frequência cada atributo
    # aparece nos melhores cromossomos? Atributos consistentemente escolhidos
    # em execuções independentes são a evidência mais forte de relevância.
    matriz_melhores = np.stack(melhores_por_experimento)
    frequencia = matriz_melhores.mean(axis=0)  # fração dos experimentos que selecionou cada atributo

    ranking = pd.DataFrame({
        'atributo': nomes_atributos,
        'frequencia_selecao': frequencia,
    }).sort_values('frequencia_selecao', ascending=False).reset_index(drop=True)
    ranking.to_csv('ranking_atributos.csv', index=False)

    print("\n" + "=" * 40)
    print("FREQUÊNCIA DE SELEÇÃO DOS ATRIBUTOS")
    print(f"(fração dos {NUM_EXPERIMENTOS} experimentos em que cada atributo foi escolhido)")
    print("=" * 40)
    print(ranking.to_string(index=False))

    # 4. Resultados finais do melhor cromossomo global
    atributos_selecionados = [
        nomes_atributos[i] for i in range(num_atributos_total)
        if melhor_cromossomo_geral[i] == 1
    ]

    print("\n" + "=" * 40)
    print("RESULTADOS FINAIS")
    print("=" * 40)
    print(f"Melhor Fitness Encontrado (validação): {melhor_fitness_geral:.4f}")
    print(f"Fitness médio entre experimentos: {np.mean(fitness_por_experimento):.4f} "
          f"(desvio {np.std(fitness_por_experimento):.4f})")
    print(f"Número de atributos selecionados: {len(atributos_selecionados)} de {num_atributos_total}")
    print(f"Atributos utilizados: {atributos_selecionados}")

    # 5. Avaliação IMPARCIAL no conjunto de teste (nunca visto pelo AG),
    # comparando o subconjunto selecionado contra a baseline com TODOS os
    # atributos — essa comparação é o objetivo declarado do trabalho
    # (impacto da redução de dimensionalidade sobre a classificação).
    print("\nAvaliando no conjunto de TESTE (dados nunca vistos)...")
    f1_teste_selecionado = treinar_e_avaliar_rede_neural(
        melhor_cromossomo_geral, X_train, y_train, X_val, y_val, num_classes,
        X_teste=X_test, y_teste=y_test
    )
    cromossomo_completo = np.ones(num_atributos_total, dtype=int)
    f1_teste_baseline = treinar_e_avaliar_rede_neural(
        cromossomo_completo, X_train, y_train, X_val, y_val, num_classes,
        X_teste=X_test, y_teste=y_test
    )

    print(f"F1 (teste) com atributos selecionados ({len(atributos_selecionados)}): {f1_teste_selecionado:.4f}")
    print(f"F1 (teste) com todos os atributos ({num_atributos_total}):     {f1_teste_baseline:.4f}")

    # Diagnóstico: matriz de confusão e F1 por classe no teste, para verificar
    # se a rede realmente discrimina as classes (e não colapsa na majoritária).
    print("\n" + "-" * 40)
    print("DIAGNÓSTICO DO MELHOR MODELO (atributos selecionados)")
    print("-" * 40)
    diagnostico_teste(melhor_cromossomo_geral, X_train, y_train, X_val, y_val,
                      X_test, y_test, num_classes, nomes_classes)

    # 6. Resumo em arquivo para compor o relatório técnico
    with open('resultados_finais.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS FINAIS — Seleção de Atributos com AG + RNA\n")
        f.write("=" * 55 + "\n")
        f.write(f"Experimentos executados: {NUM_EXPERIMENTOS}\n")
        f.write(f"Classes: {nomes_classes}\n\n")
        f.write(f"Melhor cromossomo: {melhor_cromossomo_geral.tolist()}\n")
        f.write(f"Melhor fitness (validação): {melhor_fitness_geral:.4f}\n")
        f.write(f"Fitness médio ± desvio: {np.mean(fitness_por_experimento):.4f} "
                f"± {np.std(fitness_por_experimento):.4f}\n")
        f.write(f"Atributos selecionados ({len(atributos_selecionados)}/{num_atributos_total}): "
                f"{atributos_selecionados}\n\n")
        f.write(f"F1 no TESTE com atributos selecionados: {f1_teste_selecionado:.4f}\n")
        f.write(f"F1 no TESTE com todos os atributos:     {f1_teste_baseline:.4f}\n")
    print("\nResumo salvo em 'resultados_finais.txt' e ranking em 'ranking_atributos.csv'.")

    # 7. Curva de convergência: média dos melhores nos 20 experimentos
    tamanho_maximo = max(len(h) for h in historico_experimentos)

    # Padroniza o tamanho das listas (experimentos podem parar mais cedo por estagnação)
    historico_padronizado = []
    for h in historico_experimentos:
        h_ext = list(h) + [h[-1]] * (tamanho_maximo - len(h))
        historico_padronizado.append(h_ext)

    curva_media = np.mean(historico_padronizado, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(curva_media, label=f'Média do melhor fitness ({NUM_EXPERIMENTOS} experimentos)',
             color='blue', linewidth=2)
    plt.title('Curva de Convergência do Algoritmo Genético', fontsize=14)
    plt.xlabel('Gerações', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('convergencia_ag.png', dpi=300, bbox_inches='tight')
    print("Gráfico de convergência salvo como 'convergencia_ag.png'!")
    plt.show()


if __name__ == "__main__":
    main()
