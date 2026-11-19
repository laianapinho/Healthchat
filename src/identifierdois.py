import pandas as pd

df = pd.read_csv("/home/laiana/Healthchat/src/openCHA/datasets/medDataset_processed.csv")

# acha as colunas sem depender de maiúscula/minúscula
cols = {c.lower(): c for c in df.columns}
q, a, t = cols["question"], cols["answer"], cols["qtype"]

print("Linhas:", len(df))
print("Perguntas únicas:", df[q].nunique())
print()
print(df[t].value_counts())
print()

palavras = df[a].astype(str).str.split().str.len()
print("Tamanho da resposta (palavras):")
print(palavras.describe())
print()
print("Respostas com mais de 300 palavras:",
      (palavras > 300).sum(), f"({(palavras > 300).mean():.1%})")
print("Respostas vazias:", df[a].isna().sum())