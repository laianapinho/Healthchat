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
from openCHA.dataset_tools.csv_column_detector import detect_csv_qa_columns
from openCHA.llmjudge import (
    LLMJudgeEvaluator,
    geval_chat,
    geval_arena,
    geval_csv,
)

logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
_CSS = """
/* ── Fonte e base ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
}

/* ── Header ───────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 8px;
    border: 1px solid #1e40af22;
}
.app-header h1 {
    color: #e2e8f0 !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.3px;
}
.app-header p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}

/* ── Accordion de API keys ────────────────────────────────── */
.keys-accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
}

/* ── Abas ─────────────────────────────────────────────────── */
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    border-radius: 8px 8px 0 0 !important;
    color: #64748b !important;
    border: none !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #1d4ed8 !important;
    border-bottom: 2px solid #1d4ed8 !important;
    background: #eff6ff !important;
}

/* ── Botões primários ─────────────────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.2px;
    transition: opacity 0.15s !important;
}
.gr-button-primary:hover { opacity: 0.88 !important; }

/* ── Cards de resultado (arena) ───────────────────────────── */
.model-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}

/* ── Tabelas ──────────────────────────────────────────────── */
.gr-dataframe table {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}
.gr-dataframe th {
    background: #f1f5f9 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.5px;
}

/* ── Score badges na aba manual ───────────────────────────── */
.score-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
}
.score-badge {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 6px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #1d4ed8;
    font-weight: 500;
}

/* ── Seção de resultado na aba manual ────────────────────── */
.result-section {
    border-left: 3px solid #1d4ed8;
    padding-left: 12px;
    margin: 6px 0;
}

/* ── Textarea de referência ───────────────────────────────── */
.reference-box textarea {
    background: #fefce8 !important;
    border-color: #fde68a !important;
    font-size: 0.85rem !important;
}

/* ── Status messages ──────────────────────────────────────── */
.status-ok  { color: #16a34a; font-weight: 500; }
.status-err { color: #dc2626; font-weight: 500; }
"""


class Interface:
    """
    Gradio UI com 6 abas:
      1) Chat normal
      2) Comparação Multi-LLM (Arena)
      3) Benchmark JSON
      4) Benchmark CSV + BERTScore
      5) BERTScore Manual          ← NOVA
      6) G-EVAL (LLM Juiz)
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
        with self.gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter"),
            ),
            css=_CSS,
            title="HealthChat — openCHA",
        ) as demo:

            # ── Header ────────────────────────────────────────────────────────
            gr.HTML("""
            <div class="app-header">
                <h1>🔷 HealthChat · openCHA</h1>
                <p>Chat · Arena Multi-LLM · Benchmark JSON · BERTScore CSV · BERTScore Manual · G-EVAL</p>
            </div>
            """)

            # ── API Keys ──────────────────────────────────────────────────────
            with gr.Accordion("🔑 API Keys", open=True, elem_classes="keys-accordion"):
                with gr.Row():
                    openai_key   = gr.Textbox(label="OpenAI",   type="password", placeholder="sk-...")
                    serp_key     = gr.Textbox(label="SERP",     type="password", placeholder="...")
                    gemini_key   = gr.Textbox(label="Gemini",   type="password", placeholder="AIza...")
                    deepseek_key = gr.Textbox(label="DeepSeek", type="password", placeholder="...")

            with gr.Accordion("⚙️ Configurações do Agente", open=False):
                with gr.Row():
                    use_history    = gr.Checkbox(label="Usar histórico de conversa", value=True)
                    tasks_selector = gr.CheckboxGroup(
                        choices=available_tasks, label="Tasks disponíveis", value=[])

            gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:8px 0 16px'>")

            with gr.Tabs():

                # ── ABA 1: CHAT ───────────────────────────────────────────────
                with gr.Tab("💬 Chat"):
                    chatbot = gr.Chatbot(
                        label="Conversa", bubble_full_width=False,
                        height=480, show_copy_button=True)

                    with gr.Row():
                        msg_chat = gr.Textbox(
                            placeholder="Digite sua mensagem...",
                            lines=3, show_label=False, scale=5)
                        with gr.Column(scale=1, min_width=110):
                            btn_send_chat   = gr.Button("Enviar ▶", variant="primary")
                            btn_upload_chat = gr.UploadButton(
                                "📎 Arquivo",
                                file_types=["text","pdf","image","audio","video"])

                    with gr.Row():
                        btn_clear_chat = gr.Button("🗑️ Limpar", variant="secondary")
                        gr.Markdown(
                            "<span style='font-size:11px;color:#94a3b8'>"
                            "Enter envia · Shift+Enter quebra linha</span>")

                    with gr.Accordion("⚖️ G-EVAL — Avaliar última resposta", open=False):
                        chat_gabarito = gr.Textbox(
                            label="Gabarito / resposta esperada (opcional)",
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
                            btn_geval_chat = gr.Button(
                                "⚖️ Avaliar", variant="secondary", scale=1)
                        chat_geval_result = gr.Dataframe(
                            headers=["criterio","nota"],
                            datatype=["str","number"],
                            row_count=7, col_count=(2,"fixed"),
                            label="Notas G-EVAL")
                        chat_geval_status = gr.Markdown()

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

                # ── ABA 2: ARENA ──────────────────────────────────────────────
                with gr.Tab("🏟️ Arena Multi-LLM"):
                    gr.Markdown("Compare as respostas dos modelos lado a lado.")

                    with gr.Row():
                        compare_models = gr.CheckboxGroup(
                            label="Modelos",
                            choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini"], scale=1)
                        msg_arena = gr.Textbox(
                            placeholder="Digite a pergunta...",
                            lines=4, show_label=False, scale=3)

                    with gr.Accordion("⚖️ G-EVAL — Avaliar respostas", open=False):
                        arena_gabarito = gr.Textbox(
                            label="Gabarito / resposta esperada (opcional)",
                            placeholder="Deixe vazio para avaliar sem referência...",
                            lines=2)
                        arena_geval_criteria = gr.CheckboxGroup(
                            label="Critérios",
                            choices=["corretude","completude","coerencia",
                                     "consistencia","fluencia","seguranca","aderencia"],
                            value=["corretude","completude","coerencia",
                                   "consistencia","fluencia","seguranca","aderencia"])

                    btn_send_arena = gr.Button("⚔️ Comparar modelos", variant="primary")
                    arena_status   = gr.Markdown()

                    with gr.Row(equal_height=True):
                        card_chatgpt  = gr.Markdown(elem_classes="model-card")
                        card_gemini   = gr.Markdown(elem_classes="model-card")
                        card_deepseek = gr.Markdown(elem_classes="model-card")

                    arena_geval_table = gr.Dataframe(
                        headers=["modelo","corretude","completude","coerencia",
                                 "consistencia","fluencia","seguranca","aderencia","média"],
                        datatype=["str","number","number","number",
                                  "number","number","number","number","number"],
                        row_count=3, col_count=(9,"fixed"),
                        label="G-EVAL — Comparação entre modelos",
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

                        responses = {"chatgpt": out_cg, "gemini": out_gm, "deepseek": out_ds}
                        geval_rows, table_visible = geval_arena(
                            pergunta=msg, gabarito=gabarito, criteria=criteria,
                            openai_key=openai, models=models, responses=responses)

                        yield (
                            "✅ Pronto.",
                            f"### 🟢 ChatGPT\n\n{out_cg}",
                            f"### 🔵 Gemini\n\n{out_gm}",
                            f"### 🟣 DeepSeek\n\n{out_ds}",
                            table_visible,
                            geval_rows,
                        )

                    btn_send_arena.click(fn=respond_arena,
                        inputs=[msg_arena, arena_gabarito, arena_geval_criteria,
                                openai_key, serp_key, gemini_key, deepseek_key,
                                use_history, tasks_selector, compare_models],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek,
                                 arena_geval_table, arena_geval_table])

                # ── ABA 3: BENCHMARK JSON ─────────────────────────────────────
                with gr.Tab("📁 Benchmark JSON"):
                    gr.Markdown("Envie um JSON, escolha a quantidade de questões e rode o benchmark automático.")

                    file_json     = gr.File(label="Arquivo JSON", file_types=[".json"], type="binary")
                    btn_load      = gr.Button("✅ Carregar JSON", variant="secondary")
                    load_status   = gr.Markdown()

                    with gr.Row():
                        models_bench  = gr.CheckboxGroup(
                            label="Modelos",
                            choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini"], scale=2)
                        with gr.Column(scale=1):
                            num_samples   = gr.Slider(
                                minimum=1, maximum=500, value=10, step=1,
                                label="Questões")
                            random_sample = gr.Checkbox(
                                label="🎲 Seleção aleatória", value=False)

                    btn_run      = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    bench_report = gr.Textbox(label="Relatório", lines=20)
                    bench_table  = gr.Dataframe(
                        headers=["id","expected","chatgpt","gemini","deepseek",
                                 "ok_chatgpt","ok_gemini","ok_deepseek"],
                        datatype=["str"]*8, row_count=5,
                        col_count=(8,"fixed"), wrap=True)

                    def do_load(file_obj):
                        try:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info   = info
                            stats = info["stats"]
                            return (
                                f"✅ **Carregado!** · "
                                f"Total: **{stats.get('total_items')}** · "
                                f"Tipo: **{info.get('dataset_type','').upper()}** · "
                                f"Pergunta: `{info['mapping'].get('question_field')}` · "
                                f"Resposta: `{info['mapping'].get('answer_field')}`"
                            )
                        except Exception as e:
                            logger.error(f"Erro load JSON: {e}", exc_info=True)
                            return f"❌ Erro ao carregar: {e}"

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

                # ── ABA 4: BENCHMARK CSV + BERTSCORE ─────────────────────────
                with gr.Tab("📊 BERTScore — CSV"):
                    gr.Markdown(
                        "Envie o CSV com colunas de pergunta e resposta "
                        "(ex: `Pergunta`/`Resposta`, `question`/`answer`, etc. — "
                        "detectadas automaticamente), selecione os modelos e "
                        "calcule P/R/F1 automaticamente.")

                    with gr.Row():
                        file_csv = gr.File(
                            label="Arquivo CSV",
                            file_types=[".csv"], type="filepath", scale=2)
                        models_select = gr.CheckboxGroup(
                            label="Modelos",
                            choices=["chatgpt","gemini","deepseek"],
                            value=["chatgpt","gemini","deepseek"], scale=1)

                    with gr.Row():
                        csv_num_samples = gr.Slider(
                            minimum=1, maximum=500, value=1, step=1,
                            label="Quantidade de questões", scale=2)
                        csv_random = gr.Checkbox(
                            label="🎲 Seleção aleatória", value=False, scale=1)

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

                        # ── detecção automática das colunas de pergunta/resposta ──
                        try:
                            question_col, answer_col, _ = detect_csv_qa_columns(df)
                        except ValueError as e:
                            return f"❌ {e}", []

                        n  = min(int(n_samples), len(df))
                        df = df.sample(n=n, random_state=42).reset_index(drop=True) \
                             if use_random else df.head(n)

                        questions  = df[question_col].tolist()
                        references = df[answer_col].tolist()
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
                                record = self._bs_evaluator.build_record(
                                    pergunta=query,
                                    modelo=model,
                                    resposta=model_response,
                                    reference=references[i],
                                )
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

                # ── ABA 5: BERTSCORE MANUAL (NOVA) ────────────────────────────
                with gr.Tab("🎯 BERTScore — Manual"):
                    gr.Markdown(
                        "Digite a pergunta, a resposta de referência e compare "
                        "os modelos com BERTScore em tempo real.")

                    with gr.Row():
                        with gr.Column(scale=1):
                            manual_question = gr.Textbox(
                                label="❓ Pergunta",
                                placeholder="Ex: Tenho febre há 3 dias, o que fazer?",
                                lines=4)
                            manual_reference = gr.Textbox(
                                label="📋 Resposta de referência (gabarito)",
                                placeholder="Digite aqui a resposta esperada / gabarito médico...",
                                lines=6,
                                elem_classes="reference-box")
                            manual_models = gr.CheckboxGroup(
                                label="Modelos a avaliar",
                                choices=["chatgpt","gemini","deepseek"],
                                value=["chatgpt","gemini","deepseek"])
                            btn_run_manual = gr.Button(
                                "🎯 Calcular BERTScore", variant="primary")

                        with gr.Column(scale=1):
                            gr.Markdown("### Resultados")
                            manual_status = gr.Markdown(
                                value="_Aguardando avaliação..._")

                            manual_table = gr.Dataframe(
                                headers=["Modelo","Precisão (P)","Recall (R)","F1"],
                                datatype=["str","number","number","number"],
                                row_count=3, col_count=(4,"fixed"),
                                label="BERTScore por modelo")

                            gr.Markdown("### Respostas geradas")
                            manual_resp_chatgpt  = gr.Textbox(
                                label="🟢 ChatGPT",  lines=5, interactive=False)
                            manual_resp_gemini   = gr.Textbox(
                                label="🔵 Gemini",   lines=5, interactive=False)
                            manual_resp_deepseek = gr.Textbox(
                                label="🟣 DeepSeek", lines=5, interactive=False)

                    def run_manual_bertscore(
                        question, reference, selected_models,
                        openai, serp, gemini, deepseek, use_hist, tasks
                    ):
                        # Validações
                        if not question or not question.strip():
                            return (
                                "❌ Digite uma pergunta.", [], "", "", ""
                            )
                        if not reference or not reference.strip():
                            return (
                                "❌ Digite a resposta de referência (gabarito).", [], "", "", ""
                            )
                        if not selected_models:
                            return (
                                "❌ Selecione pelo menos 1 modelo.", [], "", "", ""
                            )

                        # Gera respostas dos modelos
                        try:
                            _, updated_hist = respond(
                                question, openai, serp, gemini, deepseek,
                                [], use_hist, tasks, True, selected_models)
                            report_text = ""
                            if updated_hist and updated_hist[-1] and \
                                    len(updated_hist[-1]) == 2:
                                report_text = updated_hist[-1][1] or ""
                        except Exception as e:
                            return f"❌ Erro ao gerar respostas: {e}", [], "", "", ""

                        # Coleta respostas por modelo
                        model_responses = {}
                        for model in selected_models:
                            model_responses[model] = extract_model_response_from_report(
                                report_text, model.upper()) or ""

                        # Calcula BERTScore
                        table_rows = []
                        records    = []
                        for model in selected_models:
                            response_text = model_responses.get(model, "")
                            record = self._bs_evaluator.build_record(
                                pergunta=question,
                                modelo=model,
                                resposta=response_text,
                                reference=reference,
                            )
                            records.append(record)
                            table_rows.append([
                                model.upper(),
                                record["P"],
                                record["R"],
                                record["F1"],
                            ])

                        # Salva junto com os resultados do CSV
                        BertScoreEvaluator.save(records)

                        # Resumo
                        summary = self._bs_evaluator.summarize(
                            records, selected_models, 1)

                        # Melhor modelo
                        best = max(records, key=lambda r: r["F1"])
                        status_md = (
                            f"{summary}\n\n"
                            f"🏆 **Melhor F1:** `{best['modelo'].upper()}` "
                            f"→ **{best['F1']:.4f}**\n\n"
                            f"✅ Resultado salvo em `results_multillm.json`"
                        )

                        return (
                            status_md,
                            table_rows,
                            model_responses.get("chatgpt",  "— não selecionado —"),
                            model_responses.get("gemini",   "— não selecionado —"),
                            model_responses.get("deepseek", "— não selecionado —"),
                        )

                    btn_run_manual.click(
                        fn=run_manual_bertscore,
                        inputs=[
                            manual_question, manual_reference, manual_models,
                            openai_key, serp_key, gemini_key, deepseek_key,
                            use_history, tasks_selector,
                        ],
                        outputs=[
                            manual_status, manual_table,
                            manual_resp_chatgpt, manual_resp_gemini, manual_resp_deepseek,
                        ])

                # ── ABA 6: G-EVAL ─────────────────────────────────────────────
                with gr.Tab("🏛️ G-EVAL (LLM Juiz)"):
                    gr.Markdown(
                        "Avalia as respostas dos modelos com um LLM como juiz.\n\n"
                        "As colunas de pergunta e resposta (gabarito) do CSV são "
                        "detectadas automaticamente.")

                    with gr.Row():
                        geval_file_csv = gr.File(
                            label="Arquivo CSV",
                            file_types=[".csv"], type="filepath", scale=2)
                        geval_models = gr.CheckboxGroup(
                            label="Modelos a avaliar",
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
                            label="Modelo juiz",
                            choices=["gpt-4o-mini","gpt-4o","gpt-4-turbo"],
                            value="gpt-4o-mini", scale=1)
                        geval_criteria = gr.CheckboxGroup(
                            label="Critérios",
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

            demo.launch(
                share=share,
                server_port=server_port,
                server_name="0.0.0.0",
                show_error=True)