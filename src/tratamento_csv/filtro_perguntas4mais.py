import pandas as pd

file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas.csv"
output_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas_avaliacao4mais.csv"

df = pd.read_csv(file_path)

print(f"Total de linhas antes: {len(df)}")
print(f"Perguntas únicas antes: {df['Pergunta'].nunique()}")

# Mantém só respostas com avaliação >= 4 (gabarito validado por humano)
df_avaliadas = df[df['Avaliação'] >= 4]
print(f"Linhas com Avaliação >= 4: {len(df_avaliadas)}")

# Ordena por Avaliação e Curtidas (melhores primeiro)
df_ordenado = df_avaliadas.sort_values(by=['Avaliação', 'Curtidas'], ascending=False)

# Mantém só a melhor resposta pra cada pergunta
df_filtrado = df_ordenado.drop_duplicates(subset='Pergunta', keep='first')

# Reordena por ID
df_filtrado = df_filtrado.sort_values(by='ID').reset_index(drop=True)

print(f"Total de linhas depois: {len(df_filtrado)}")
print(f"Perguntas únicas depois: {df_filtrado['Pergunta'].nunique()}")

# Salva em um novo arquivo (não sobrescreve o original)
df_filtrado.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")