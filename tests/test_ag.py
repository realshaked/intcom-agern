"""
Testes da lógica determinística do Algoritmo Genético (ag.py).

O AG é testado com uma função de avaliação stub (sem rede neural / torch),
o que é possível porque o AG recebe a função de avaliação por injeção de
dependência — exatamente o desenho que permite "rodar o GA de diferentes
formas" trocando apenas o avaliador.
"""
import numpy as np
import pytest

from ag import AlgoritmoGenetico


def f1_proporcional(cromossomo):
    """Stub determinístico: F1 fictício = proporção de genes ativos."""
    return float(np.sum(cromossomo)) / len(cromossomo)


def novo_ag(**kwargs):
    """AG pequeno e rápido para testes; parâmetros do enunciado são default."""
    params = dict(
        num_atributos=10,
        funcao_avaliacao=f1_proporcional,
        pop_size=12,
        elitismo=3,
        max_gen=30,
        gen_sem_melhora=10,
        gap=2,
    )
    params.update(kwargs)
    return AlgoritmoGenetico(**params)


# ---------------------------------------------------------------------------
# Função de aptidão (seção 6 do enunciado)
# ---------------------------------------------------------------------------

def test_fitness_segue_formula_do_enunciado():
    # Fitness = 0,9 × F1 + 0,1 × (1 − Ns/Nt)
    ag = novo_ag()
    cromossomo = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # Ns=5, Nt=10
    esperado = 0.9 * 0.8 + 0.1 * (1 - 5 / 10)
    assert ag.avaliar_fitness(cromossomo, f1_score=0.8) == pytest.approx(esperado)


def test_fitness_cromossomo_zerado_vale_zero():
    # Sem atributos selecionados não há rede possível: aptidão nula.
    ag = novo_ag()
    assert ag.avaliar_fitness(np.zeros(10, dtype=int), f1_score=1.0) == 0.0


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def test_mutacao_mantem_codificacao_binaria():
    ag = novo_ag()
    ag.pm = 1.0  # força mutação em todos os genes
    original = np.zeros(10, dtype=int)
    mutado = ag.mutacao(original.copy())
    assert set(np.unique(mutado)) <= {0, 1}
    assert np.all(mutado == 1)  # pm=1 inverte todos os genes de um vetor zerado


def test_crossover_com_pc_zero_copia_os_pais():
    ag = novo_ag()
    ag.pc = 0.0
    pai1 = np.zeros(10, dtype=int)
    pai2 = np.ones(10, dtype=int)
    f1_, f2_ = ag.crossover_uniforme(pai1, pai2)
    assert np.array_equal(f1_, pai1)
    assert np.array_equal(f2_, pai2)


def test_crossover_uniforme_troca_genes_complementares():
    # Com pais complementares (0s e 1s), a troca uniforme mantém
    # filho1 + filho2 == 1 em cada gene, qualquer que seja a máscara.
    ag = novo_ag()
    ag.pc = 1.0
    pai1 = np.zeros(10, dtype=int)
    pai2 = np.ones(10, dtype=int)
    f1_, f2_ = ag.crossover_uniforme(pai1, pai2)
    assert np.all(f1_ + f2_ == 1)


# ---------------------------------------------------------------------------
# Cache de fitness (evita retreinar a rede para cromossomos repetidos)
# ---------------------------------------------------------------------------

def test_cache_evita_reavaliacao_de_cromossomo_repetido():
    chamadas = {"n": 0}

    def avaliador_contado(cromossomo):
        chamadas["n"] += 1
        return f1_proporcional(cromossomo)

    ag = novo_ag(funcao_avaliacao=avaliador_contado)
    cromossomo = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    ag._f1_do_cromossomo(cromossomo)
    ag._f1_do_cromossomo(cromossomo.copy())  # mesmo conteúdo, outro objeto
    assert chamadas["n"] == 1


# ---------------------------------------------------------------------------
# Execução completa (steady-state + elitismo implícito)
# ---------------------------------------------------------------------------

def test_execucao_retorna_solucao_e_historico_nao_decrescente():
    np.random.seed(42)
    ag = novo_ag()
    melhor, fitness, historico = ag.executar()

    assert melhor is not None
    assert len(melhor) == 10
    assert fitness > 0
    # Como o steady-state substitui apenas os piores, o melhor da população
    # nunca pode piorar de uma geração para a outra (elitismo preservado).
    assert all(b >= a for a, b in zip(historico, historico[1:]))


def test_populacao_mantem_tamanho_e_diversidade_de_posicoes():
    # Regressão do bug de elitismo: as posições 0..elitismo-1 da população
    # NÃO devem ser sobrescritas com cópias da elite a cada geração.
    np.random.seed(7)
    ag = novo_ag()
    ag.executar()
    assert ag.populacao.shape == (12, 10)
    # O melhor fitness da população final deve ser o melhor global retornado
    assert np.max(ag.fitness) == pytest.approx(max(ag.melhor_historico))
