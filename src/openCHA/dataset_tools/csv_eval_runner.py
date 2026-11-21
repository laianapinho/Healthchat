"""
Executor único usado pelas abas de BERTScore / BioBERT / PubMedBERT.

Cada aba só chama run_csv_eval(...) dizendo qual avaliador e qual idioma usar.

Alteração: se o CSV não tiver coluna de resposta (mode="no_reference"), o executor
só gera as respostas dos modelos e salva em CSV. Não calcula P/R/F1, porque não há
referência para comparar.
"""
import logging
from datetime import datetime

import pandas as pd

from .csv_prepare_pt import (
    prepare_pt_items, extract_decision, extract_explanation,
    best_record, decision_metrics,
)

logger = logging.getLogger(__name__)


def run_csv_eval(
    file_path, selected_models, n_samples, use_random,
    openai, serp, gemini, deepseek, use_hist, tasks,
    *, respond, extract_fn, evaluator, lang,
):
    if file_path is None:
        return "❌ Nenhum arquivo enviado.", []
    if not selected_models:
        return "❌ Selecione pelo menos 1 modelo.", []
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"❌ Erro ao ler CSV: {e}", []

    # ── prepara as questões (PubMedQA, LiveQA, MedQuAD ou outro) ──
    try:
        data = prepare_pt_items(df, int(n_samples), use_random, lang=lang)
    except ValueError as e:
        return f"❌ {e}", []

    n = len(data["prompts"])
    if n == 0:
        return "❌ Nenhuma linha válida (pergunta e resposta vazias).", []
    records, table_rows = [], []

    for i, prompt in enumerate(data["prompts"]):
        try:
            _, updated_hist = respond(
                prompt, openai, serp, gemini, deepseek,
                [], use_hist, tasks, True, selected_models)
            report_text = ""
            if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                report_text = updated_hist[-1][1] or ""
        except Exception as e:
            logger.error(f"Erro na questão {i}: {e}", exc_info=True)
            report_text = ""

        for model in selected_models:
            model_response = extract_fn(report_text, model.upper()) or ""

            # ── Sem coluna de resposta: só registra a resposta do modelo ──
            if data["mode"] == "no_reference":
                records.append({
                    "pergunta": data["questions"][i],
                    "modelo": model,
                    "resposta": model_response,
                })
                table_rows.append([
                    data["questions"][i], model, model_response,
                    None, None, None,
                ])
                continue

            if data["mode"] == "pubmedqa":
                pred = extract_decision(model_response)
                resposta = extract_explanation(model_response)
            else:
                pred, resposta = None, model_response

            record = best_record(
                evaluator,
                pergunta=data["questions"][i],
                modelo=model,
                resposta=resposta,
                references=data["references"][i],
            )
            if data["mode"] == "pubmedqa":
                record["final_decision"] = data["labels"][i]
                record["predicted"] = pred

            records.append(record)
            table_rows.append([
                data["questions"][i], model, model_response,
                record["P"], record["R"], record["F1"],
            ])

    if data["mode"] == "no_reference":
        out_path = f"respostas_sem_referencia_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
        return (
            "ℹ️ O CSV não tem coluna de resposta, então não há referência para o "
            "BERTScore. Foram geradas apenas as respostas dos modelos "
            f"({n} perguntas).\n\n✅ Respostas salvas em `{out_path}`."
        ), table_rows

    saved_path = evaluator.save(records)
    summary = evaluator.summarize(records, selected_models, n)
    if data["mode"] == "pubmedqa":
        summary += "\n\n" + decision_metrics(records, selected_models)
    return f"{summary}\n\n✅ Resultados salvos em `{saved_path}`.", table_rows