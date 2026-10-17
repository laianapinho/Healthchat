import json
import pandas as pd

with open("/home/laiana/Healthchat/src/openCHA/datasets/ori_pqal.json", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for pubid, item in data.items():
    rows.append({
        "pubid": pubid,
        "question": item["QUESTION"],
        "context": " ".join(item["CONTEXTS"]),
        "long_answer": item["LONG_ANSWER"],
        "final_decision": item["final_decision"],
    })

pd.DataFrame(rows).to_csv("/home/laiana/Healthchat/src/openCHA/datasets/pubmedqa_labeled.csv", index=False)