import pandas as pd

file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_amostra_120.csv"
output_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_psiquiatria.csv"

ESPECIALIDADE_ALVO = "psiquiatria-58065"   # troque pelo nome exato da especialidade

df = pd.read_csv(file_path)

print(f"Total de linhas no arquivo: {len(df)}")
print()
print("Especialidades disponíveis (nome exato pra usar no filtro):")
print(df['Especialidade'].unique())
print()

df_especialidade = df[df['Especialidade'] == ESPECIALIDADE_ALVO].reset_index(drop=True)

print(f"Linhas encontradas para '{ESPECIALIDADE_ALVO}': {len(df_especialidade)}")

df_especialidade.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")