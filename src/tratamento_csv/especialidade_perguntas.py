import pandas as pd

file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas_avaliacao4mais.csv"

df = pd.read_csv(file_path)

print(f"Total de linhas: {len(df)}")
print(f"Total de especialidades: {df['Especialidade'].nunique()}")
print()

contagem = df['Especialidade'].value_counts()

print("Quantidade de perguntas por especialidade:")
print(contagem)

print()
print(f"Especialidade com mais perguntas: {contagem.idxmax()} ({contagem.max()})")
print(f"Especialidade com menos perguntas: {contagem.idxmin()} ({contagem.min()})")
print(f"Média de perguntas por especialidade: {contagem.mean():.1f}")
print(f"Mediana de perguntas por especialidade: {contagem.median():.1f}")