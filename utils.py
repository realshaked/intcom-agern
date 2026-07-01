import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def pre_processar_dados(caminho_arquivo, target_col='label_cid'):
    """
    Executa as 5 etapas de pré-processamento exigidas no trabalho.
    """
    print(f"Lendo o arquivo: {caminho_arquivo}...")
    df = pd.read_excel(caminho_arquivo)

    # Converter a string 'NULL' que veio do banco de dados para NaN real do Pandas
    df.replace('NULL', np.nan, inplace=True)


    # ETAPA 4: Análise Exploratória (Feita antes de alterar muito os dados)
    
    print("\n" + "="*40)
    print(" ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*40)
    print(f"Total de registros (linhas): {df.shape[0]}")
    print(f"Total de atributos (colunas): {df.shape[1]}")
    print(f"\nDistribuição das classes ({target_col}):")
    print(df[target_col].value_counts())
    print("-" * 40)

    

    # ETAPA 1: Tratamento de valores ausentes

    # Remove colunas onde mais de 50% dos dados são nulos (lixo não ajuda a rede neural)
    limite = len(df) * 0.5
    df.dropna(thresh=limite, axis=1, inplace=True)

    # Preenche os nulos que sobraram
    for col in df.columns:
        # Verifica se a coluna é estritamente numérica
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            # Se for texto, string, object, categoria, etc., preenche com a moda
            df[col] = df[col].fillna(df[col].mode()[0])


    # ETAPA 2: Remoção de registros inconsistentes

    # Removeremos linhas onde a classe alvo (o que queremos prever) não existe
    df.dropna(subset=[target_col], inplace=True)



    # ETAPA 3: Conversão de atributos categóricos

    # CUIDADO COM DATA LEAKAGE: Colunas que são literalmente a resposta do câncer.
    # Adicionadas CAUSABAS_O e CB_PRE que também vazam códigos de doenças.
    colunas_vazadas_ou_inuteis = [
        'DTOBITO', 'DTNASC', 'CAUSABAS', 'causabas_categoria', 
        'causabas_subcategoria', 'LINHAA', 'LINHAB', 'LINHAC', 'LINHAD', 'LINHAII',
        'CAUSABAS_O', 'CB_PRE'
    ]
    df.drop(columns=[c for c in colunas_vazadas_ou_inuteis if c in df.columns], inplace=True, errors='ignore')

    # Instancia o conversor
    le = LabelEncoder()

    # Separa o alvo (y) e converte classes de texto (ex: C53) para números (0, 1, 2)
    y_raw = df[target_col]
    y = le.fit_transform(y_raw)
    
    # Salva o número de classes geradas para configurar a camada de saída da Rede Neural
    num_classes = len(np.unique(y))
    
    # Separa os atributos (X)
    X_df = df.drop(columns=[target_col])

    # Converte qualquer coluna de texto restante (ex: 'S'/'N', 'AC') para números
    # Usando is_numeric_dtype para não cair na armadilha das novas versões do Pandas
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    # ETAPA 5: Aplicação da normalização linear Min-Max

    # Aplica a fórmula exigida: x' = (x - xmin) / (xmax - xmin)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df)
    
    nomes_atributos = X_df.columns.tolist()
    
    print("\n✅ Pré-processamento concluído!")
    print(f"Atributos restantes após limpeza: {X.shape[1]}")
    
    return X, y, nomes_atributos, num_classes

def separar_conjuntos(X, y):
    """
    Divide os dados em: 70% Treino, 15% Validação, 15% Teste
    """
    # Primeiro separa 70% para treino e 30% para o resto
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Divide os 30% restantes no meio (15% validação, 15% teste)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    return X_train, y_train, X_val, y_val, X_test, y_test