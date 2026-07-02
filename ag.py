import numpy as np
import random


class AlgoritmoGenetico:
    """
    Steady-State Genetic Algorithm para seleção de atributos (seção 5 do enunciado).

    O AG não conhece a rede neural: ele recebe `funcao_avaliacao`, que mapeia
    um cromossomo binário -> F1-Score. Essa injeção de dependência permite
    rodar o MESMO AG de diferentes formas (outra rede, outra métrica, stub de
    teste) trocando apenas o avaliador — e é o que torna a lógica testável
    sem depender de torch.
    """

    def __init__(self, num_atributos, funcao_avaliacao, pop_size=150, pc=0.85,
                 elitismo=10, max_gen=200, gen_sem_melhora=20, gap=2, usar_cache=True):
        self.L = num_atributos
        self.funcao_avaliacao = funcao_avaliacao
        self.pop_size = pop_size          # 150 cromossomos [cite: 37]
        self.pc = pc                      # Probabilidade de crossover de 0,85 [cite: 39]
        self.pm = 1.0 / self.L            # Probabilidade de mutação 1/L [cite: 40, 42]
        self.elitismo = elitismo          # 10 melhores preservados [cite: 43]
        self.max_gen = max_gen            # Máximo de 200 gerações [cite: 46]
        self.gen_sem_melhora = gen_sem_melhora  # Parada após 20 gerações sem melhora [cite: 47]
        self.gap = gap                    # Steady-State com Gap = 2 [cite: 45]

        # Cache de fitness: no steady-state, cromossomos idênticos reaparecem
        # com frequência; sem cache a rede neural seria retreinada à toa.
        self._cache = {} if usar_cache else None

        self.populacao = []
        self.fitness = []
        self.melhor_historico = []

    def inicializar_populacao(self):
        # Gene = 1 (selecionado), Gene = 0 (não selecionado) [cite: 29]
        self.populacao = np.random.randint(2, size=(self.pop_size, self.L))

    def _f1_do_cromossomo(self, cromossomo):
        """Obtém o F1 via avaliador externo, com memoização por conteúdo."""
        if self._cache is None:
            return self.funcao_avaliacao(cromossomo)
        chave = np.asarray(cromossomo, dtype=np.uint8).tobytes()
        if chave not in self._cache:
            self._cache[chave] = self.funcao_avaliacao(cromossomo)
        return self._cache[chave]

    def avaliar_fitness(self, cromossomo, f1_score):
        Ns = np.sum(cromossomo)
        Nt = self.L

        # Cromossomo sem nenhum atributo não gera rede válida: aptidão nula.
        if Ns == 0:
            return 0.0

        # Função de aptidão: Fitness = 0,9 × F1-Score + 0,1 × (1 - Ns/Nt) [cite: 52]
        return 0.9 * f1_score + 0.1 * (1 - (Ns / Nt))

    def normalizar_fitness(self, fitness_array):
        # Normalização linear dos valores de fitness para escalonamento [cite: 58],
        # convertida em distribuição de probabilidade para a seleção por roleta.
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

    def executar(self, verbose=True):
        self.inicializar_populacao()
        self.fitness = np.zeros(self.pop_size)
        self.melhor_historico = []

        if verbose:
            print("Avaliando população inicial...")
        for i in range(self.pop_size):
            f1 = self._f1_do_cromossomo(self.populacao[i])
            self.fitness[i] = self.avaliar_fitness(self.populacao[i], f1)

        melhor_solucao_global = None
        melhor_fitness_global = -1
        geracoes_sem_melhora = 0

        for geracao in range(self.max_gen):
            indices_ordenados = np.argsort(self.fitness)[::-1]

            melhor_da_geracao = self.fitness[indices_ordenados[0]]
            self.melhor_historico.append(melhor_da_geracao)

            if melhor_da_geracao > melhor_fitness_global:
                melhor_fitness_global = melhor_da_geracao
                melhor_solucao_global = self.populacao[indices_ordenados[0]].copy()
                geracoes_sem_melhora = 0
            else:
                geracoes_sem_melhora += 1

            if geracoes_sem_melhora >= self.gen_sem_melhora:
                if verbose:
                    print(f"Parada antecipada na geração {geracao} por estagnação.")
                break

            fitness_norm = self.normalizar_fitness(self.fitness)

            # Steady-State: seleciona 2 pais, cruza, muta e substitui apenas os
            # `gap` piores indivíduos. Como os 10 melhores nunca estão entre os
            # piores, o elitismo é garantido POR CONSTRUÇÃO — não é necessário
            # (nem correto) recopiar a elite para posições fixas da população,
            # o que duplicaria indivíduos e destruiria a diversidade.
            idx_pai1 = self.selecao_roleta(fitness_norm)
            idx_pai2 = self.selecao_roleta(fitness_norm)

            filho1, filho2 = self.crossover_uniforme(self.populacao[idx_pai1], self.populacao[idx_pai2])
            filho1 = self.mutacao(filho1)
            filho2 = self.mutacao(filho2)

            fit_filho1 = self.avaliar_fitness(filho1, self._f1_do_cromossomo(filho1))
            fit_filho2 = self.avaliar_fitness(filho2, self._f1_do_cromossomo(filho2))

            piores_indices = indices_ordenados[-self.gap:]

            self.populacao[piores_indices[0]] = filho1
            self.fitness[piores_indices[0]] = fit_filho1

            self.populacao[piores_indices[1]] = filho2
            self.fitness[piores_indices[1]] = fit_filho2

            if verbose:
                print(f"Geração {geracao} | Melhor Fitness: {melhor_fitness_global:.4f}")

        return melhor_solucao_global, melhor_fitness_global, self.melhor_historico
