import pandas as pd

# Define o caminho do arquivo CSV que será validado.
# Por enquanto, estamos usando o caminho local do projeto.
#file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas.csv"
file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostasv2.csv"

# Cria uma função responsável por validar os dados de um arquivo CSV.
# O parâmetro file_path representa o caminho do arquivo que será lido.
def validate_data(file_path):
    # Lê o arquivo CSV usando pandas e armazena os dados em um DataFrame.
    # Um DataFrame pode ser entendido como uma tabela dentro do Python.
    df = pd.read_csv(file_path)

    # Conta o total de linhas existentes no arquivo CSV.
    total_linhas = len(df)

    # Seleciona as primeiras linhas do DataFrame para visualização inicial.
    primeiras_linhas = df.head()

    # Conta quantas linhas duplicadas existem no DataFrame.
    total_duplicados = df.duplicated().sum()

    # Conta quantos campos vazios existem em cada coluna.
    campos_vazios_por_coluna = df.isnull().sum()

    # Exibe o total de linhas do arquivo.
    print("Total de linhas:")
    print(total_linhas)

    # Exibe as primeiras linhas do arquivo para conferência visual.
    print("\nPrimeiras linhas:")
    print(primeiras_linhas)

    # Exibe o total de linhas duplicadas encontradas.
    print("\nDuplicados:")
    print(total_duplicados)

    # Exibe a quantidade de campos vazios por coluna.
    print("\nCampos vazios por coluna:")
    print(campos_vazios_por_coluna)


# Chama a função de validação, passando o caminho do arquivo CSV.
validate_data(file_path)