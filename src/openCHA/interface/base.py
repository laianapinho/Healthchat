# openCHA/interface/base.py
import gradio as gr
import logging
from typing import List, Tuple, Optional, Any
import pandas as pd
from bert_score import score
from datetime import datetime
import json
import os

from openCHA.benchmark_ui_helpers import (
    load_dataset_from_gradio_file,
    run_json_benchmark,
    extract_model_response_from_report,
)
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


class Interface:
    """
    Gradio UI com 4 abas:
      1) Chat normal
      2) Comparação Multi-LLM (Arena)
      3) Benchmark JSON
      4) Benchmark CSV + BERTScore
    """

    def __init__(self):
        self.gr = gr
        self._benchmark_loader = None
        self._benchmark_info = None
        logger.info("Interface inicializada")

    def compute_bertscore(self, hypotheses: list, references: list):
        P, R, F1 = score(hypotheses, references, lang="pt", batch_size=16)
        return P.tolist(), R.tolist(), F1.tolist()

    def save_bertscore_results(self, results: list, file_path="results_multillm.json"):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        else:
            old_data = []
        old_data.extend(results)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, indent=2, ensure_ascii=False)
        return file_path

    def prepare_interface(
        self,
        respond,
        reset,
        upload_meta,
        available_tasks: List[str],
        share: bool = False,
        server_port: int = 7860,
    ):
        with self.gr.Blocks(theme=gr.themes.Soft(), title="openCHA") as demo:

            gr.Markdown("""
            # 🔷 openCHA
            **Modos:** Chat normal | Arena Multi-LLM | Benchmark JSON | Benchmark CSV + BERTScore
            """)

            with gr.Accordion("🔑 Configuração de API Keys", open=True):
                with gr.Row():
                    openai_key   = gr.Textbox(label="🟢 OpenAI API Key",   type="password")
                    serp_key     = gr.Textbox(label="🔍 SERP API Key",     type="password")
                with gr.Row():
                    gemini_key   = gr.Textbox(label="🔵 Gemini API Key",   type="password")
                    deepseek_key = gr.Textbox(label="🟣 DeepSeek API Key", type="password")

            with gr.Accordion("⚙️ Configurações do Agente", open=False):
                with gr.Row():
                    use_history    = gr.Checkbox(label="💬 Usar histórico", value=True)
                    tasks_selector = gr.CheckboxGroup(choices=available_tasks, label="🛠️ Tasks", value=[])

            gr.Markdown("---")

            with gr.Tabs():

                # ── ABA 1: CHAT ──────────────────────────────────────────
                with gr.Tab("💬 Chat normal"):
                    chatbot = gr.Chatbot(label="Conversa", bubble_full_width=False, height=520, show_copy_button=True)

                    with gr.Row():
                        msg_chat = gr.Textbox(placeholder="Digite sua mensagem...", lines=3, show_label=False)
                        with gr.Column(scale=1, min_width=120):
                            btn_send_chat   = gr.Button("🚀 Enviar", variant="primary")
                            btn_upload_chat = gr.UploadButton("📎 Arquivo", file_types=["text","pdf","image","audio","video"])

                    with gr.Row():
                        btn_clear_chat = gr.Button("🗑️ Limpar conversa", variant="secondary")
                        gr.Markdown("<span class='small-note'>Enter envia | Shift+Enter quebra linha</span>")

                    state_chat_history = gr.State([])

                    def render_history(chat_history):
                        return [(u, a) for (u, a) in chat_history if a is not None]

                    def reset_wrapper_chat():
                        try:
                            reset()
                            return [], []
                        except Exception as e:
                            logger.error(f"Erro ao resetar: {e}")
                            return [], []

                    def respond_chat_wrapper(msg, openai, serp, gemini, deepseek, chat_hist, use_hist, tasks):
                        if not msg or not msg.strip():
                            chat_hist.append((msg, "⚠️ Digite uma mensagem."))
                            return "", chat_hist
                        chat_hist.append((msg, "⏳ Processando..."))
                        yield "", chat_hist
                        try:
                            empty_msg, updated = respond(
                                msg, openai, serp, gemini, deepseek,
                                chat_hist[:-1], use_hist, tasks, False
                            )
                            yield empty_msg, updated
                        except Exception as e:
                            logger.error(f"Erro respond_chat_wrapper: {e}", exc_info=True)
                            chat_hist[-1] = (msg, f"❌ Erro: {str(e)}")
                            yield "", chat_hist

                    msg_chat.submit(fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key, state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history])
                    btn_send_chat.click(fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key, state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history])
                    state_chat_history.change(fn=render_history, inputs=[state_chat_history], outputs=[chatbot])
                    btn_clear_chat.click(fn=reset_wrapper_chat, inputs=[], outputs=[state_chat_history, chatbot])
                    btn_upload_chat.upload(fn=upload_meta, inputs=[state_chat_history, btn_upload_chat], outputs=[state_chat_history])

                # ── ABA 2: ARENA ─────────────────────────────────────────
                with gr.Tab("🏟️ Comparação Multi-LLM"):
                    gr.Markdown("**Escreva um prompt e veja respostas lado a lado.**")

                    compare_models = gr.CheckboxGroup(
                        label="🤖 Modelos", choices=["chatgpt","gemini","deepseek"], value=["chatgpt","gemini"])
                    msg_arena      = gr.Textbox(placeholder="Digite a pergunta...", lines=3, show_label=False)
                    btn_send_arena = gr.Button("⚔️ Comparar", variant="primary")
                    arena_status   = gr.Markdown()

                    with gr.Row(equal_height=True):
                        card_chatgpt  = gr.Markdown()
                        card_gemini   = gr.Markdown()
                        card_deepseek = gr.Markdown()

                    def respond_arena(msg, openai, serp, gemini, deepseek, use_hist, tasks, models):
                        if not msg or not msg.strip():
                            return "⚠️ Digite uma mensagem.", "", "", ""
                        if not models:
                            return "⚠️ Selecione pelo menos 1 modelo.", "", "", ""
                        yield f"⏳ Comparando {', '.join(models)}...", "", "", ""
                        _, updated_hist = respond(msg, openai, serp, gemini, deepseek, [], use_hist, tasks, True, models)
                        report_text = ""
                        if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                            report_text = updated_hist[-1][1] or ""
                        def pick(k):
                            return extract_model_response_from_report(report_text, k.upper()) or "—"
                        yield (
                            "✅ Pronto.",
                            f"### ChatGPT\n\n{pick('chatgpt')  if 'chatgpt'  in models else '—'}",
                            f"### Gemini\n\n{pick('gemini')    if 'gemini'   in models else '—'}",
                            f"### DeepSeek\n\n{pick('deepseek') if 'deepseek' in models else '—'}",
                        )

                    btn_send_arena.click(fn=respond_arena,
                        inputs=[msg_arena, openai_key, serp_key, gemini_key, deepseek_key, use_history, tasks_selector, compare_models],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek])

                # ── ABA 3: BENCHMARK JSON ────────────────────────────────
                with gr.Tab("📁 Benchmark JSON"):
                    gr.Markdown("**Upload de JSON** → escolha **quantidade de questões** → **rodar automático**")

                    file_json     = gr.File(label="Envie o JSON", file_types=[".json"], type="binary")
                    btn_load      = gr.Button("✅ Carregar JSON", variant="primary")
                    load_status   = gr.Markdown()
                    models_bench  = gr.CheckboxGroup(label="🤖 Modelos", choices=["chatgpt","gemini","deepseek"], value=["chatgpt","gemini"])
                    num_samples   = gr.Slider(minimum=1, maximum=500, value=10, step=1, label="Quantidade de questões")
                    random_sample = gr.Checkbox(label="🎲 Seleção aleatória", value=False)
                    btn_run       = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    bench_report  = gr.Textbox(label="Relatório", lines=25)
                    bench_table   = gr.Dataframe(
                        headers=["id","expected","chatgpt","gemini","deepseek","ok_chatgpt","ok_gemini","ok_deepseek"],
                        datatype=["str"]*8, row_count=5, col_count=(8,"fixed"), wrap=True)

                    def do_load(file_obj):
                        try:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info   = info
                            stats = info["stats"]
                            return (
                                f"✅ **Carregado!**\n\n"
                                f"• Total: **{stats.get('total_items')}**\n"
                                f"• Tipo: **{info.get('dataset_type','').upper()}**\n"
                                f"• Campos: pergunta=`{info['mapping'].get('question_field')}`, "
                                f"resposta=`{info['mapping'].get('answer_field')}`\n"
                            )
                        except Exception as e:
                            logger.error(f"Erro load JSON: {e}", exc_info=True)
                            return f"❌ Erro ao carregar JSON: {e}"

                    def do_run(file_obj, models, n, rnd, openai, serp, gemini, deepseek, use_hist, tasks):
                        if self._benchmark_loader is None:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info   = info

                        def run_single_question(question, use_multi_llm=True, compare_models=None):
                            _empty, updated_hist = respond(question, openai, serp, gemini, deepseek, [], use_hist, tasks, True, compare_models)
                            if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                                return updated_hist[-1][1] or ""
                            return ""

                        report, rows = run_json_benchmark(
                            run_single_question=run_single_question,
                            loader=self._benchmark_loader,
                            models=models, num_samples=int(n), show_per_question=True)

                        evaluator = BenchmarkEvaluator()
                        def ok_flag(dataset_type, expected, answer):
                            if (dataset_type or "").lower().strip() != "closed":
                                return ""
                            return "✅" if evaluator.extract_answer(answer) == str(expected).strip().lower() else "❌"

                        table_rows = []
                        for r in rows:
                            ds_type  = str(r.get("dataset_type",""))
                            expected = r.get("expected","")
                            cg, gm, ds = r.get("chatgpt",""), r.get("gemini",""), r.get("deepseek","")
                            table_rows.append([
                                str(r.get("id","")), str(expected), str(cg), str(gm), str(ds),
                                ok_flag(ds_type, expected, str(cg)),
                                ok_flag(ds_type, expected, str(gm)),
                                ok_flag(ds_type, expected, str(ds)),
                            ])
                        return report, table_rows

                    btn_load.click(fn=do_load, inputs=[file_json], outputs=[load_status])
                    btn_run.click(fn=do_run,
                        inputs=[file_json, models_bench, num_samples, random_sample,
                                openai_key, serp_key, gemini_key, deepseek_key, use_history, tasks_selector],
                        outputs=[bench_report, bench_table])

                # ── ABA 4: BENCHMARK CSV + BERTSCORE ────────────────────
                with gr.Tab("📊 Benchmark CSV + BERTScore"):
                    gr.Markdown("**Upload CSV → calcula P/R/F1 (BERTScore) por modelo**\n\nCSV deve ter colunas `Pergunta` e `Resposta`.")

                    with gr.Row():
                        file_csv = gr.File(label="📄 Envie o CSV", file_types=[".csv"], type="filepath", scale=2)
                        models_select = gr.CheckboxGroup(
                            label="🤖 Modelos", choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini","deepseek"], scale=1)

                    with gr.Row():
                        csv_num_samples = gr.Slider(minimum=1, maximum=500, value=1, step=1,
                                                    label="Quantidade de questões", scale=2)
                        csv_random      = gr.Checkbox(label="🎲 Seleção aleatória", value=False, scale=1)

                    btn_run_csv = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    csv_status  = gr.Markdown()
                    csv_table   = gr.Dataframe(
                        headers=["pergunta","modelo","resposta","P","R","F1"],
                        datatype=["str","str","str","number","number","number"],
                        row_count=5, col_count=(6,"fixed"), wrap=True, label="Resultados BERTScore")

                    def run_csv_benchmark(file_path, selected_models, n_samples, use_random,
                                          openai, serp, gemini, deepseek, use_hist, tasks):
                        if file_path is None:
                            return "❌ Nenhum arquivo enviado.", []
                        if not selected_models:
                            return "❌ Selecione pelo menos 1 modelo.", []

                        try:
                            # type="filepath" entrega string diretamente — sem .name
                            df = pd.read_csv(file_path)
                        except Exception as e:
                            return f"❌ Erro ao ler CSV: {e}", []

                        if not {"Pergunta","Resposta"}.issubset(df.columns):
                            return "❌ CSV deve conter as colunas 'Pergunta' e 'Resposta'.", []

                        # ── limita amostras ──────────────────────────────
                        n = min(int(n_samples), len(df))
                        df = df.sample(n=n, random_state=42).reset_index(drop=True) if use_random else df.head(n)
                        # ────────────────────────────────────────────────

                        questions  = df["Pergunta"].tolist()
                        references = df["Resposta"].tolist()
                        results_list, table_rows = [], []

                        for i, query in enumerate(questions):
                            # chama o agente e pega o relatório completo
                            try:
                                _, updated_hist = respond(
                                    query, openai, serp, gemini, deepseek,
                                    [], use_hist, tasks, True, selected_models)
                                report_text = ""
                                if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                                    report_text = updated_hist[-1][1] or ""
                            except Exception as e:
                                logger.error(f"Erro na questão {i}: {e}", exc_info=True)
                                report_text = ""

                            for model in selected_models:
                                # ✅ extrai SÓ a resposta do modelo — não o relatório inteiro
                                model_response = extract_model_response_from_report(report_text, model.upper()) or ""

                                try:
                                    P, R, F1 = self.compute_bertscore([model_response], [references[i]])
                                    p_val, r_val, f1_val = P[0], R[0], F1[0]
                                except Exception as e:
                                    logger.error(f"Erro BERTScore q{i} {model}: {e}")
                                    p_val = r_val = f1_val = 0.0

                                print(f"[{model.upper()}] {query[:60]}")
                                print(f"Resposta: {model_response[:120]}")
                                print(f"P={p_val:.3f}  R={r_val:.3f}  F1={f1_val:.3f}")
                                print("-" * 50)

                                record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "pergunta": query, "modelo": model,
                                    "resposta": model_response,          # ✅ resposta individual
                                    "P": round(p_val, 4),
                                    "R": round(r_val, 4),
                                    "F1": round(f1_val, 4),
                                }
                                results_list.append(record)
                                table_rows.append([query, model, model_response,
                                                   round(p_val,4), round(r_val,4), round(f1_val,4)])

                        saved_path = self.save_bertscore_results(results_list)

                        summary_lines = [f"### 📈 Médias por modelo ({n} questão/ões)\n"]
                        for model in selected_models:
                            mrs = [r for r in results_list if r["modelo"] == model]
                            if mrs:
                                summary_lines.append(
                                    f"- **{model.upper()}** → "
                                    f"P={sum(r['P'] for r in mrs)/len(mrs):.4f} | "
                                    f"R={sum(r['R'] for r in mrs)/len(mrs):.4f} | "
                                    f"F1={sum(r['F1'] for r in mrs)/len(mrs):.4f}"
                                )
                        summary_lines.append(f"\n✅ Resultados salvos em `{saved_path}`.")
                        return "\n".join(summary_lines), table_rows

                    btn_run_csv.click(
                        fn=run_csv_benchmark,
                        inputs=[file_csv, models_select, csv_num_samples, csv_random,
                                openai_key, serp_key, gemini_key, deepseek_key,
                                use_history, tasks_selector],
                        outputs=[csv_status, csv_table])

            demo.launch(share=share, server_port=server_port, server_name="0.0.0.0", show_error=True)