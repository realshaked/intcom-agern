import numpy as np
import random
from redes import treinar_e_avaliar_rede_neural

class AlgoritmoGenetico:
    def __init__(self, num_atributos, pop_size=150, pc=0.85, elitismo=10, max_gen=200, gen_sem_melhora=20, gap=2):
        self.L = num_atributos
        self.pop_size = pop_size # 150 cromossomos [cite: 37]
        self.pc = pc # Probabilidade de crossover de 0,85 [cite: 39]
        self.pm = 1.0 / self.L # Probabilidade de mutação 1/L [cite: 40, 42]
        self.elitismo = elitismo # 10 melhores preservados [cite: 43]
        self.max_gen = max_gen # Máximo de 200 gerações [cite: 46]
        self.gen_sem_melhora = gen_sem_melhora # Parada após 20 gerações sem melhora [cite: 47]
        self.gap = gap # Steady-State com Gap = 2 [cite: 45]
        
        self.populacao = []
        self.fitness = []
        self.melhor_historico = []

    def inicializar_populacao(self):
        # Gene = 1 (selecionado), Gene = 0 (não selecionado) [cite: 29]
        self.populacao = np.random.randint(2, size=(self.pop_size, self.L))

    def avaliar_fitness(self, cromossomo, f1_score):
        Ns = np.sum(cromossomo)
        Nt = self.L
        
        if Ns == 0:
            return 0.0
            
        # Função de aptidão: Fitness = 0,9 × F1-Score + 0,1 × (1 - Ns/Nt) [cite: 52]
        fitness = 0.9 * f1_score + 0.1 * (1 - (Ns / Nt))
        return fitness

    def normalizar_fitness(self, fitness_array):
        # Normalização linear Min-Max dos valores de fitness [cite: 58]
        f_min = np.min(fitness_array)
        f_max = np.max(fitness_array)
        if f_max == f_min:
            return np.ones_like(fitness_array) / len(fitness_array)
        
        norm = (fitness_array - f_min) / (f_max - f_min)
        return norm / np.sum(norm)

    def crossover_uniforme(self, pai1, pai2):
        # Operador de Crossover Uniforme [cite: 38]
        filho1, filho2 = pai1.copy(), pai2.copy()
        if random.random() < self.pc:
            mascara = np.random.randint(2, size=self.L)
            for i in range(self.L):
                if mascara[i] == 1:
                    filho1[i], filho2[i] = pai2[i], pai1[i]
        return filho1, filho2

    def mutacao(self, cromossomo):
        for i in range(self.L):
            if random.random() < self.pm:
                cromossomo[i] = 1 - cromossomo[i]
        return cromossomo

    def selecao_roleta(self, fitness_norm):
        r = random.random()
        acumulado = 0.0
        for i, prob in enumerate(fitness_norm):
            acumulado += prob
            if r <= acumulado:
                return i
        return len(fitness_norm) - 1

    def executar(self, X_train, y_train, X_val, y_val, num_classes):
        self.inicializar_populacao()
        self.fitness = np.zeros(self.pop_size)
        
        print("Avaliando população inicial...")
        for i in range(self.pop_size):
            f1_score = treinar_e_avaliar_rede_neural(
                self.populacao[i], X_train, y_train, X_val, y_val, num_classes
            )
            self.fitness[i] = self.avaliar_fitness(self.populacao[i], f1_score)

        melhor_solucao_global = None
        melhor_fitness_global = -1
        geracoes_sem_melhora = 0

        for geracao in range(self.max_gen):
            indices_ordenados = np.argsort(self.fitness)[::-1]
            elite = self.populacao[indices_ordenados[:self.elitismo]].copy()
            fitness_elite = self.fitness[indices_ordenados[:self.elitismo]].copy()
            
            melhor_da_geracao = fitness_elite[0]
            self.melhor_historico.append(melhor_da_geracao)

            if melhor_da_geracao > melhor_fitness_global:
                melhor_fitness_global = melhor_da_geracao
                melhor_solucao_global = elite[0].copy()
                geracoes_sem_melhora = 0
            else:
                geracoes_sem_melhora += 1

            if geracoes_sem_melhora >= self.gen_sem_melhora:
                print(f"Parada antecipada na geração {geracao} por estagnação.")
                break

            fitness_norm = self.normalizar_fitness(self.fitness)

            # Steady-State: Seleciona 2, cruza, muta e substitui os 2 piores
            idx_pai1 = self.selecao_roleta(fitness_norm)
            idx_pai2 = self.selecao_roleta(fitness_norm)
            
            filho1, filho2 = self.crossover_uniforme(self.populacao[idx_pai1], self.populacao[idx_pai2])
            filho1 = self.mutacao(filho1)
            filho2 = self.mutacao(filho2)
            
            f1_filho1 = treinar_e_avaliar_rede_neural(filho1, X_train, y_train, X_val, y_val, num_classes)
            f1_filho2 = treinar_e_avaliar_rede_neural(filho2, X_train, y_train, X_val, y_val, num_classes)
            
            fit_filho1 = self.avaliar_fitness(filho1, f1_filho1)
            fit_filho2 = self.avaliar_fitness(filho2, f1_filho2)
            
            piores_indices = indices_ordenados[-self.gap:]
            
            self.populacao[piores_indices[0]] = filho1
            self.fitness[piores_indices[0]] = fit_filho1
            
            self.populacao[piores_indices[1]] = filho2
            self.fitness[piores_indices[1]] = fit_filho2

            self.populacao[:self.elitismo] = elite
            self.fitness[:self.elitismo] = fitness_elite
            
            print(f"Geração {geracao} | Melhor Fitness: {melhor_fitness_global:.4f}")

        return melhor_solucao_global, melhor_fitness_global, self.melhor_historico