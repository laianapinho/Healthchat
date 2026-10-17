import pandas as pd

df = pd.read_csv("liveqa_por_pergunta.csv")

df = df.rename(columns={
    "paraphrase": "question",
    "answers": "answer",
})

df.to_csv("liveqa_agente.csv", index=False)