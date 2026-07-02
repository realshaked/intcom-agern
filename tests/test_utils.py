"""
Testes do pré-processamento (utils.py) usando um DataFrame sintético.

Cobrem principalmente a correção do bug de ordem das etapas: registros sem
classe alvo devem ser REMOVIDOS antes da imputação — jamais receber a moda
como rótulo falso.
"""
import numpy as np
import pandas as pd
import pytest

from utils import pre_processar_df, separar_conjuntos


def df_sintetico():
    return pd.DataFrame({
        "idade": [30.0, 40.0, np.nan, 50.0, 60.0, 35.0],
        "estado": ["RJ", "SP", "RJ", None, "RJ", "SP"],
        "renda": [1000.0, 2000.0, 1500.0, np.nan, 3000.0, 1200.0],
        # linha 2 não tem diagnóstico -> deve ser descartada, não imputada
        "label_cid": ["C53", "C54", None, "C53", "C54", "C53"],
    })


def test_registros_sem_classe_alvo_sao_removidos_e_nao_imputados():
    X, y, nomes, num_classes, nomes_classes = pre_processar_df(
        df_sintetico(), target_col="label_cid"
    )
    # 6 linhas no total, 1 sem alvo -> 5 devem sobrar
    assert X.shape[0] == 5
    assert len(y) == 5


def test_atributos_normalizados_no_intervalo_0_1_e_sem_nulos():
    X, *_ = pre_processar_df(df_sintetico(), target_col="label_cid")
    assert not np.isnan(X).any()
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_string_NULL_tratada_como_valor_ausente():
    df = df_sintetico()
    # Colunas vindas do banco com 'NULL' chegam do Excel como dtype object
    df["renda"] = pd.Series(["NULL", 2000.0, 1500.0, 2500.0, 3000.0, 1200.0],
                            dtype=object)
    X, *_ = pre_processar_df(df, target_col="label_cid")
    assert not np.isnan(X).any()


def test_classes_mapeadas_e_contadas():
    _, y, _, num_classes, nomes_classes = pre_processar_df(
        df_sintetico(), target_col="label_cid"
    )
    assert num_classes == 2
    # O mapeamento das classes deve ser preservado para o relatório
    assert sorted(nomes_classes) == ["C53", "C54"]
    assert set(np.unique(y)) == {0, 1}


def test_colunas_com_maioria_de_nulos_sao_descartadas():
    df = df_sintetico()
    df["coluna_lixo"] = [np.nan, np.nan, np.nan, np.nan, 1.0, np.nan]
    _, _, nomes, _, _ = pre_processar_df(df, target_col="label_cid")
    assert "coluna_lixo" not in nomes


def test_registros_duplicados_sao_removidos():
    df = pd.concat([df_sintetico(), df_sintetico().iloc[[0]]], ignore_index=True)
    X, y, *_ = pre_processar_df(df, target_col="label_cid")
    assert X.shape[0] == 5  # duplicata da linha 0 descartada


def test_separacao_70_15_15_estratificada():
    rng = np.random.default_rng(0)
    X = rng.random((100, 4))
    y = np.array([0] * 50 + [1] * 50)
    X_train, y_train, X_val, y_val, X_test, y_test = separar_conjuntos(X, y)
    assert len(y_train) == 70
    assert len(y_val) == 15
    assert len(y_test) == 15
    # Estratificação: proporção de classes preservada em cada conjunto
    assert np.mean(y_train) == pytest.approx(0.5)
    assert np.mean(y_test) == pytest.approx(0.5, abs=0.1)
