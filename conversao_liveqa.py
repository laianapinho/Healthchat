import pandas as pd
import xml.etree.ElementTree as ET

caminho = "/home/laiana/Healthchat/src/openCHA/datasets/TREC-2017-LiveQA-Medical-Test.xml"

def limpa(texto):
    return " ".join(texto.split()) if texto else ""

root = ET.parse(caminho).getroot()

rows = []
for q in root.findall("NLM-QUESTION"):
    orig = q.find("Original-Question")
    ann = q.find("ANNOTATIONS")
    answers = q.find("ReferenceAnswers")
    lista = list(answers) if answers is not None else []  # RefAnswer ou ReferenceAnswer

    rows.append({
        "qid": q.get("qid"),
        "subject": limpa(orig.findtext("SUBJECT")),
        "original_message": limpa(orig.findtext("MESSAGE")),
        "question": limpa(q.findtext("NIST-PARAPHRASE")),
        "focus": "; ".join(limpa(f.text) for f in ann.findall("FOCUS")),
        "type": "; ".join(limpa(t.text) for t in ann.findall("TYPE")),
        "keyword": "; ".join(limpa(k.text) for k in ann.findall("KEYWORD")),
        "n_answers": len(lista),
        "answer": " ||| ".join(limpa(a.findtext("ANSWER")) for a in lista),
        "answer_urls": " ||| ".join(limpa(a.findtext("AnswerURL")) for a in lista),
    })

pd.DataFrame(rows).to_csv("liveqa_agente.csv", index=False)
print(len(rows), "perguntas salvas")