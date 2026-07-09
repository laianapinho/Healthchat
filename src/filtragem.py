import pandas as pd

file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas.csv"
output_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas_filtrado.csv"

df = pd.read_csv(file_path)

print(f"Total de linhas antes: {len(df)}")
print(f"Perguntas únicas: {df['Pergunta'].nunique()}")

# Ordena por Avaliação e Curtidas (melhores primeiro)
df_ordenado = df.sort_values(by=['Avaliação', 'Curtidas'], ascending=False)

# Mantém só a melhor resposta pra cada pergunta
df_filtrado = df_ordenado.drop_duplicates(subset='Pergunta', keep='first')

# Reordena por ID
df_filtrado = df_filtrado.sort_values(by='ID').reset_index(drop=True)

print(f"Total de linhas depois: {len(df_filtrado)}")
print(f"Linhas removidas: {len(df) - len(df_filtrado)}")

# Salva em um novo arquivo (não sobrescreve o original)
df_filtrado.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")