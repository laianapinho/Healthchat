import pandas as pd
df = pd.read_csv("/home/laiana/Healthchat/src/openCHA/datasets/medpt.csv")
print("Linhas:", len(df))
print("Perguntas únicas:", df["question"].nunique())
print(df["question_type"].value_counts())