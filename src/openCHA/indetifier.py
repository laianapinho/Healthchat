import pandas as pd
from openCHA.dataset_tools.csv_column_detector import detect_csv_qa_columns

df = pd.read_csv("/home/laiana/Healthchat/src/openCHA/datasets/pubmedqa_labeled.csv")
print(df.columns.tolist())

q, a, info = detect_csv_qa_columns(df)
print("Pergunta:", q)
print("Resposta:", a)