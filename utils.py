import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def pre_processar_dados(caminho_arquivo, target_col='label_cid'):
    """
    Lê o arquivo Excel e delega ao núcleo de pré-processamento.
    Mantida separada de `pre_processar_df` para que a lógica seja testável
    com DataFrames sintéticos, sem depender do arquivo real.
    """
    print(f"Lendo o arquivo: {caminho_arquivo}...")
    df = pd.read_excel(caminho_arquivo)
    return pre_processar_df(df, target_col=target_col)


def pre_processar_df(df, target_col='label_cid'):
    """
    Executa as 5 etapas de pré-processamento exigidas no trabalho (seção 3).

    Retorna: X (normalizado), y (codificado), nomes_atributos, num_classes,
             nomes_classes (mapeamento índice -> rótulo original, ex: 0 -> 'C53').
    """
    df = df.copy()

    # Converter a string 'NULL' que veio do banco de dados para NaN real do Pandas
    df.replace('NULL', np.nan, inplace=True)

    # ETAPA 4: Análise Exploratória (feita antes de alterar os dados)
    print("\n" + "=" * 40)
    print(" ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 40)
    print(f"Total de registros (linhas): {df.shape[0]}")
    print(f"Total de atributos (colunas): {df.shape[1]}")
    print(f"\nDistribuição das classes ({target_col}):")
    print(df[target_col].value_counts(dropna=False))
    percentual_nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    print("\nAtributos com valores ausentes (%):")
    print(percentual_nulos[percentual_nulos > 0].round(1).to_string())
    print("-" * 40)

    # ETAPA 2: Remoção de registros inconsistentes.
    # IMPORTANTE: isto DEVE acontecer ANTES da imputação — um registro sem a
    # classe alvo não pode ganhar a moda como rótulo falso; ele é descartado.
    df.dropna(subset=[target_col], inplace=True)
    df.drop_duplicates(inplace=True)

    # ETAPA 1: Tratamento de valores ausentes
    # Remove colunas onde mais de 50% dos dados são nulos (lixo não ajuda a rede)
    limite = len(df) * 0.5
    df.dropna(thresh=limite, axis=1, inplace=True)

    # Preenche os nulos restantes — apenas nos ATRIBUTOS, nunca no alvo
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ETAPA 3: Conversão de atributos categóricos

    # CUIDADO COM DATA LEAKAGE: colunas que são literalmente a resposta do câncer.
    # Adicionadas CAUSABAS_O e CB_PRE que também vazam códigos de doenças.
    colunas_vazadas_ou_inuteis = [
        'DTOBITO', 'DTNASC', 'CAUSABAS', 'causabas_categoria',
        'causabas_subcategoria', 'LINHAA', 'LINHAB', 'LINHAC', 'LINHAD', 'LINHAII',
        'CAUSABAS_O', 'CB_PRE'
    ]
    df.drop(columns=[c for c in colunas_vazadas_ou_inuteis if c in df.columns],
            inplace=True, errors='ignore')

    # Encoder EXCLUSIVO do alvo: preserva o mapeamento número -> classe (ex: 0 -> 'C53')
    # para que o relatório consiga nomear as classes depois.
    le_y = LabelEncoder()
    y = le_y.fit_transform(df[target_col].astype(str))
    nomes_classes = le_y.classes_.tolist()
    num_classes = len(nomes_classes)

    # Separa os atributos (X)
    X_df = df.drop(columns=[target_col])

    # Converte cada coluna de texto restante (ex: 'S'/'N', 'AC') com um encoder próprio
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    # ETAPA 5: Aplicação da normalização linear Min-Max
    # Aplica a fórmula exigida: x' = (x - xmin) / (xmax - xmin)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df)

    nomes_atributos = X_df.columns.tolist()

    print("\nPré-processamento concluído!")
    print(f"Registros após limpeza: {X.shape[0]}")
    print(f"Atributos restantes após limpeza: {X.shape[1]}")

    return X, y, nomes_atributos, num_classes, nomes_classes


def separar_conjuntos(X, y):
    """
    Divide os dados em: 70% Treino, 15% Validação, 15% Teste (seção 8).
    """
    # Primeiro separa 70% para treino e 30% para o resto
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Divide os 30% restantes no meio (15% validação, 15% teste)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
