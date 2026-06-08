# openCHA/interface/base.py
import gradio as gr
import logging
import pandas as pd
from typing import List, Tuple, Optional, Any

from openCHA.benchmark_ui_helpers import (
    load_dataset_from_gradio_file,
    run_json_benchmark,
    extract_model_response_from_report,
)
from openCHA.benchmark_evaluator import BenchmarkEvaluator
from openCHA.bertscore_evaluator import BertScoreEvaluator
from openCHA.llmjudge import (
    LLMJudgeEvaluator,
    geval_chat,
    geval_arena,
    geval_csv,
)

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
        self._benchmark_info   = None
        self._bs_evaluator     = BertScoreEvaluator(lang="pt", batch_size=16)
        logger.info("Interface inicializada")

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
                    chatbot = gr.Chatbot(label="Conversa", bubble_full_width=False,
                                        height=520, show_copy_button=True)

                    with gr.Row():
                        msg_chat = gr.Textbox(placeholder="Digite sua mensagem...",
                                             lines=3, show_label=False)
                        with gr.Column(scale=1, min_width=120):
                            btn_send_chat   = gr.Button("🚀 Enviar", variant="primary")
                            btn_upload_chat = gr.UploadButton("📎 Arquivo",
                                file_types=["text","pdf","image","audio","video"])

                    with gr.Row():
                        btn_clear_chat = gr.Button("🗑️ Limpar conversa", variant="secondary")
                        gr.Markdown("<span style='font-size:12px;opacity:.75'>Enter envia | Shift+Enter quebra linha</span>")

                    # ── G-EVAL no chat ───────────────────────────────────
                    with gr.Accordion("⚖️ G-EVAL — Avaliar resposta com LLM juiz", open=False):
                        chat_gabarito = gr.Textbox(
                            label="📋 Gabarito / resposta esperada (opcional)",
                            placeholder="Deixe vazio para avaliar sem referência...",
                            lines=3)
                        with gr.Row():
                            chat_geval_criteria = gr.CheckboxGroup(
                                label="Critérios",
                                choices=["corretude","completude","coerencia",
                                         "consistencia","fluencia","seguranca","aderencia"],
                                value=["corretude","completude","coerencia",
                                       "consistencia","fluencia","seguranca","aderencia"],
                                scale=2)
                            btn_geval_chat = gr.Button("⚖️ Avaliar última resposta",
                                                       variant="secondary", scale=1)
                        chat_geval_result = gr.Dataframe(
                            headers=["criterio","nota"],
                            datatype=["str","number"],
                            row_count=7, col_count=(2,"fixed"),
                            label="Notas G-EVAL")
                        chat_geval_status = gr.Markdown()
                    # ────────────────────────────────────────────────────

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

                    def respond_chat_wrapper(msg, openai, serp, gemini, deepseek,
                                             chat_hist, use_hist, tasks):
                        if not msg or not msg.strip():
                            chat_hist.append((msg, "⚠️ Digite uma mensagem."))
                            return "", chat_hist
                        chat_hist.append((msg, "⏳ Processando..."))
                        yield "", chat_hist
                        try:
                            empty_msg, updated = respond(
                                msg, openai, serp, gemini, deepseek,
                                chat_hist[:-1], use_hist, tasks, False)
                            yield empty_msg, updated
                        except Exception as e:
                            logger.error(f"Erro respond_chat_wrapper: {e}", exc_info=True)
                            chat_hist[-1] = (msg, f"❌ Erro: {str(e)}")
                            yield "", chat_hist

                    # geval_chat definido em openCHA/llmjudge.py

                    msg_chat.submit(fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key,
                                state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history])
                    btn_send_chat.click(fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key,
                                state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history])
                    state_chat_history.change(fn=render_history,
                        inputs=[state_chat_history], outputs=[chatbot])
                    btn_clear_chat.click(fn=reset_wrapper_chat,
                        inputs=[], outputs=[state_chat_history, chatbot])
                    btn_upload_chat.upload(fn=upload_meta,
                        inputs=[state_chat_history, btn_upload_chat],
                        outputs=[state_chat_history])
                    btn_geval_chat.click(fn=geval_chat,
                        inputs=[state_chat_history, chat_gabarito,
                                chat_geval_criteria, openai_key],
                        outputs=[chat_geval_result, chat_geval_status])

                # ── ABA 2: ARENA ─────────────────────────────────────────
                with gr.Tab("🏟️ Comparação Multi-LLM"):
                    gr.Markdown("**Escreva um prompt e veja respostas lado a lado.**")

                    compare_models = gr.CheckboxGroup(label="🤖 Modelos",
                        choices=["chatgpt","gemini","deepseek"], value=["chatgpt","gemini"])
                    msg_arena      = gr.Textbox(placeholder="Digite a pergunta...",
                                               lines=3, show_label=False)

                    # ── G-EVAL na arena ──────────────────────────────────
                    with gr.Accordion("⚖️ G-EVAL — Avaliar respostas com LLM juiz", open=False):
                        arena_gabarito = gr.Textbox(
                            label="📋 Gabarito / resposta esperada (opcional)",
                            placeholder="Deixe vazio para avaliar sem referência...",
                            lines=2)
                        arena_geval_criteria = gr.CheckboxGroup(
                            label="Critérios",
                            choices=["corretude","completude","coerencia",
                                     "consistencia","fluencia","seguranca","aderencia"],
                            value=["corretude","completude","coerencia",
                                   "consistencia","fluencia","seguranca","aderencia"])
                    # ────────────────────────────────────────────────────

                    btn_send_arena = gr.Button("⚔️ Comparar", variant="primary")
                    arena_status   = gr.Markdown()

                    with gr.Row(equal_height=True):
                        card_chatgpt  = gr.Markdown()
                        card_gemini   = gr.Markdown()
                        card_deepseek = gr.Markdown()

                    # tabela G-EVAL arena (visível após comparar)
                    arena_geval_table = gr.Dataframe(
                        headers=["modelo","corretude","completude","coerencia",
                                 "consistencia","fluencia","seguranca","aderencia","média"],
                        datatype=["str","number","number","number",
                                  "number","number","number","number","number"],
                        row_count=3, col_count=(9,"fixed"),
                        label="📊 G-EVAL — Comparação entre modelos",
                        visible=False)

                    def respond_arena(msg, gabarito, criteria, openai, serp, gemini,
                                      deepseek, use_hist, tasks, models):
                        if not msg or not msg.strip():
                            return "⚠️ Digite uma mensagem.", "", "", "", gr.update(visible=False), []
                        if not models:
                            return "⚠️ Selecione pelo menos 1 modelo.", "", "", "", gr.update(visible=False), []

                        yield f"⏳ Comparando {', '.join(models)}...", "", "", "", gr.update(visible=False), []

                        _, updated_hist = respond(msg, openai, serp, gemini, deepseek,
                                                  [], use_hist, tasks, True, models)
                        report_text = ""
                        if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                            report_text = updated_hist[-1][1] or ""

                        def pick(k):
                            return extract_model_response_from_report(report_text, k.upper()) or "—"

                        out_cg = pick("chatgpt")  if "chatgpt"  in models else "—"
                        out_gm = pick("gemini")   if "gemini"   in models else "—"
                        out_ds = pick("deepseek") if "deepseek" in models else "—"

                        # ── G-EVAL via llmjudge.py ────────────────
                        responses     = {"chatgpt": out_cg, "gemini": out_gm, "deepseek": out_ds}
                        geval_rows, table_visible = geval_arena(
                            pergunta=msg, gabarito=gabarito, criteria=criteria,
                            openai_key=openai, models=models, responses=responses)
                        # ────────────────────────────────────────────────

                        yield (
                            "✅ Pronto.",
                            f"### ChatGPT\n\n{out_cg}",
                            f"### Gemini\n\n{out_gm}",
                            f"### DeepSeek\n\n{out_ds}",
                            table_visible,
                            geval_rows,
                        )

                    btn_send_arena.click(fn=respond_arena,
                        inputs=[msg_arena, arena_gabarito, arena_geval_criteria,
                                openai_key, serp_key, gemini_key, deepseek_key,
                                use_history, tasks_selector, compare_models],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek,
                                 arena_geval_table, arena_geval_table])

                # ── ABA 3: BENCHMARK JSON ────────────────────────────────
                with gr.Tab("📁 Benchmark JSON"):
                    gr.Markdown("**Upload de JSON** → escolha **quantidade de questões** → **rodar automático**")

                    file_json     = gr.File(label="Envie o JSON", file_types=[".json"], type="binary")
                    btn_load      = gr.Button("✅ Carregar JSON", variant="primary")
                    load_status   = gr.Markdown()
                    models_bench  = gr.CheckboxGroup(label="🤖 Modelos",
                        choices=["chatgpt","gemini","deepseek"], value=["chatgpt","gemini"])
                    num_samples   = gr.Slider(minimum=1, maximum=500, value=10, step=1,
                                             label="Quantidade de questões")
                    random_sample = gr.Checkbox(label="🎲 Seleção aleatória", value=False)
                    btn_run       = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    bench_report  = gr.Textbox(label="Relatório", lines=25)
                    bench_table   = gr.Dataframe(
                        headers=["id","expected","chatgpt","gemini","deepseek",
                                 "ok_chatgpt","ok_gemini","ok_deepseek"],
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

                    def do_run(file_obj, models, n, rnd, openai, serp, gemini, deepseek,
                               use_hist, tasks):
                        if self._benchmark_loader is None:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info   = info

                        def run_single_question(question, use_multi_llm=True,
                                                compare_models=None):
                            _empty, updated_hist = respond(
                                question, openai, serp, gemini, deepseek,
                                [], use_hist, tasks, True, compare_models)
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
                            return "✅" if evaluator.extract_answer(answer) == \
                                          str(expected).strip().lower() else "❌"

                        table_rows = []
                        for r in rows:
                            ds_type  = str(r.get("dataset_type",""))
                            expected = r.get("expected","")
                            cg, gm, ds = r.get("chatgpt",""), r.get("gemini",""), r.get("deepseek","")
                            table_rows.append([
                                str(r.get("id","")), str(expected),
                                str(cg), str(gm), str(ds),
                                ok_flag(ds_type, expected, str(cg)),
                                ok_flag(ds_type, expected, str(gm)),
                                ok_flag(ds_type, expected, str(ds)),
                            ])
                        return report, table_rows

                    btn_load.click(fn=do_load, inputs=[file_json], outputs=[load_status])
                    btn_run.click(fn=do_run,
                        inputs=[file_json, models_bench, num_samples, random_sample,
                                openai_key, serp_key, gemini_key, deepseek_key,
                                use_history, tasks_selector],
                        outputs=[bench_report, bench_table])

                # ── ABA 4: BENCHMARK CSV + BERTSCORE ────────────────────
                with gr.Tab("📊 Benchmark CSV + BERTScore"):
                    gr.Markdown(
                        "**Upload CSV → calcula P/R/F1 (BERTScore) por modelo**\n\n"
                        "CSV deve ter colunas `Pergunta` e `Resposta`.")

                    with gr.Row():
                        file_csv = gr.File(label="📄 Envie o CSV",
                                          file_types=[".csv"], type="filepath", scale=2)
                        models_select = gr.CheckboxGroup(label="🤖 Modelos",
                            choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini","deepseek"], scale=1)

                    with gr.Row():
                        csv_num_samples = gr.Slider(minimum=1, maximum=500, value=1, step=1,
                                                    label="Quantidade de questões", scale=2)
                        csv_random      = gr.Checkbox(label="🎲 Seleção aleatória",
                                                     value=False, scale=1)

                    btn_run_csv = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    csv_status  = gr.Markdown()
                    csv_table   = gr.Dataframe(
                        headers=["pergunta","modelo","resposta","P","R","F1"],
                        datatype=["str","str","str","number","number","number"],
                        row_count=5, col_count=(6,"fixed"), wrap=True,
                        label="Resultados BERTScore")

                    def run_csv_benchmark(file_path, selected_models, n_samples, use_random,
                                          openai, serp, gemini, deepseek, use_hist, tasks):
                        if file_path is None:
                            return "❌ Nenhum arquivo enviado.", []
                        if not selected_models:
                            return "❌ Selecione pelo menos 1 modelo.", []

                        try:
                            df = pd.read_csv(file_path)
                        except Exception as e:
                            return f"❌ Erro ao ler CSV: {e}", []

                        if not {"Pergunta","Resposta"}.issubset(df.columns):
                            return "❌ CSV deve conter as colunas 'Pergunta' e 'Resposta'.", []

                        n  = min(int(n_samples), len(df))
                        df = df.sample(n=n, random_state=42).reset_index(drop=True) \
                             if use_random else df.head(n)

                        questions  = df["Pergunta"].tolist()
                        references = df["Resposta"].tolist()
                        records, table_rows = [], []

                        for i, query in enumerate(questions):
                            try:
                                _, updated_hist = respond(
                                    query, openai, serp, gemini, deepseek,
                                    [], use_hist, tasks, True, selected_models)
                                report_text = ""
                                if updated_hist and updated_hist[-1] and \
                                        len(updated_hist[-1]) == 2:
                                    report_text = updated_hist[-1][1] or ""
                            except Exception as e:
                                logger.error(f"Erro na questão {i}: {e}", exc_info=True)
                                report_text = ""

                            for model in selected_models:
                                model_response = extract_model_response_from_report(
                                    report_text, model.upper()) or ""

                                # ── BERTScore via módulo dedicado ────────
                                record = self._bs_evaluator.build_record(
                                    pergunta=query,
                                    modelo=model,
                                    resposta=model_response,
                                    reference=references[i],
                                )
                                # ────────────────────────────────────────

                                print(f"[{model.upper()}] {query[:60]}")
                                print(f"Resposta: {model_response[:120]}")
                                print(f"P={record['P']:.3f}  R={record['R']:.3f}  "
                                      f"F1={record['F1']:.3f}")
                                print("-" * 50)

                                records.append(record)
                                table_rows.append([
                                    query, model, model_response,
                                    record["P"], record["R"], record["F1"],
                                ])

                        saved_path = BertScoreEvaluator.save(records)
                        summary    = self._bs_evaluator.summarize(records, selected_models, n)
                        return f"{summary}\n\n✅ Resultados salvos em `{saved_path}`.", table_rows

                    btn_run_csv.click(
                        fn=run_csv_benchmark,
                        inputs=[file_csv, models_select, csv_num_samples, csv_random,
                                openai_key, serp_key, gemini_key, deepseek_key,
                                use_history, tasks_selector],
                        outputs=[csv_status, csv_table])


                # ── ABA 5: G-EVAL (LLM como juiz) ───────────────────────
                with gr.Tab("🏛️ G-EVAL (LLM Juiz)"):
                    gr.Markdown(
                        "**Upload CSV → avalia respostas com LLM como juiz (G-EVAL)**\n\n"
                        "CSV deve ter colunas `Pergunta` e `Resposta` (gabarito).\n\n"
                        "Critérios: Corretude · Completude · Coerência · Consistência · Fluência · Segurança · Aderência"
                    )

                    with gr.Row():
                        geval_file_csv = gr.File(
                            label="📄 Envie o CSV",
                            file_types=[".csv"], type="filepath", scale=2)
                        geval_models = gr.CheckboxGroup(
                            label="🤖 Modelos a avaliar",
                            choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini","deepseek"], scale=1)

                    with gr.Row():
                        geval_num_samples = gr.Slider(
                            minimum=1, maximum=100, value=1, step=1,
                            label="Quantidade de questões", scale=2)
                        geval_random = gr.Checkbox(
                            label="🎲 Seleção aleatória", value=False, scale=1)

                    with gr.Row():
                        geval_judge_model = gr.Dropdown(
                            label="🧑\u200d⚖️ Modelo juiz",
                            choices=["gpt-4o-mini","gpt-4o","gpt-4-turbo"],
                            value="gpt-4o-mini", scale=1)
                        geval_criteria = gr.CheckboxGroup(
                            label="📋 Critérios",
                            choices=["corretude","completude","coerencia",
                                     "consistencia","fluencia","seguranca","aderencia"],
                            value=["corretude","completude","coerencia",
                                   "consistencia","fluencia","seguranca","aderencia"],
                            scale=2)

                    btn_run_geval = gr.Button("⚖️ Avaliar com G-EVAL", variant="primary")
                    geval_status  = gr.Markdown()
                    geval_table   = gr.Dataframe(
                        headers=["pergunta","modelo","corretude","completude","coerencia",
                                 "consistencia","fluencia","seguranca","aderencia","media"],
                        datatype=["str","str","number","number","number",
                                  "number","number","number","number","number"],
                        row_count=5, col_count=(10,"fixed"), wrap=True,
                        label="Resultados G-EVAL")

                    # run_geval_benchmark definido em openCHA/llmjudge.py
                    def run_geval_benchmark(
                        file_path, selected_models, n_samples, use_random,
                        judge_model, selected_criteria,
                        openai, serp, gemini, deepseek, use_hist, tasks
                    ):
                        return geval_csv(
                            file_path=file_path,
                            selected_models=selected_models,
                            n_samples=n_samples, use_random=use_random,
                            judge_model=judge_model,
                            selected_criteria=selected_criteria,
                            openai_key=openai, serp_key=serp,
                            gemini_key=gemini, deepseek_key=deepseek,
                            use_hist=use_hist, tasks=tasks,
                            respond_fn=respond,
                            extract_fn=extract_model_response_from_report,
                        )

                    btn_run_geval.click(
                        fn=run_geval_benchmark,
                        inputs=[
                            geval_file_csv, geval_models, geval_num_samples, geval_random,
                            geval_judge_model, geval_criteria,
                            openai_key, serp_key, gemini_key, deepseek_key,
                            use_history, tasks_selector,
                        ],
                        outputs=[geval_status, geval_table])


            demo.launch(share=share, server_port=server_port,
                        server_name="0.0.0.0", show_error=True)