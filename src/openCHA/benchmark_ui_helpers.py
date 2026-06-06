# openCHA/benchmark_ui_helpers.py
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openCHA.dataset_tools import GenericDatasetLoader, MetricsSelector
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# 1) UPLOAD / LOADER
# =============================================================================

def _read_gradio_file(file_obj: Any) -> bytes:
    if file_obj is None:
        raise ValueError("Nenhum arquivo recebido")
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if hasattr(file_obj, "read") and callable(file_obj.read):
        raw = file_obj.read()
        if isinstance(raw, str):
            return raw.encode("utf-8", errors="ignore")
        return raw
    if hasattr(file_obj, "name"):
        p = Path(str(file_obj.name))
        if p.exists():
            return p.read_bytes()
    raise TypeError(f"Tipo de arquivo não suportado: {type(file_obj)}")


def load_dataset_from_gradio_file(file_obj: Any) -> Tuple[GenericDatasetLoader, Dict[str, Any]]:
    raw_bytes = _read_gradio_file(file_obj)
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("utf-8", errors="ignore")

    loader = GenericDatasetLoader()
    loader.load_from_json(content)
    stats = loader.get_stats()

    info = {
        "mapping": loader.mapping,
        "stats": stats,
        "dataset_type": stats.get("dataset_type"),
        "total_items": stats.get("total_items"),
    }

    return loader, info


# =============================================================================
# 2) EXTRAÇÃO DE RESPOSTA DO RELATÓRIO
# =============================================================================

def extract_model_response_from_report(full_report: str, model_name: str) -> str:
    """
    Extrai APENAS o texto da resposta de um modelo específico,
    sem incluir o bloco de avaliação (📊 Avaliação:) que vem depois.

    Suporta:
    - Formato compacto: "📝 CHATGPT  → ..."
    - Formato bloco:    "🤖 CHATGPT ... 📝 Resposta: ... 📊 Avaliação: ..."
    """
    if not full_report:
        return ""

    m_upper = model_name.upper()

    # 1) Linha compacta: "📝 CHATGPT → texto"
    compact_pat = rf"📝\s*{re.escape(m_upper)}\s*→\s*(.*)"
    m = re.search(compact_pat, full_report)
    if m:
        return (m.group(1) or "").strip()

    # 2) Bloco: seção entre "🤖 MODELO" e o próximo "🤖 ..." ou fim do texto
    block_pat = rf"🤖\s*{re.escape(m_upper)}(.*?)(?=🤖\s*[A-Z]+|\Z)"
    b = re.search(block_pat, full_report, flags=re.DOTALL)
    if not b:
        return ""

    block = b.group(1)

    marker = "📝 Resposta:"
    idx = block.find(marker)
    if idx != -1:
        text = block[idx + len(marker):]
        # ── CORREÇÃO: corta antes do bloco de avaliação ──────────────
        eval_idx = text.find("📊 Avaliação:")
        if eval_idx != -1:
            text = text[:eval_idx]
        # ─────────────────────────────────────────────────────────────
        return text.strip()

    return block.strip()


# =============================================================================
# 3) BENCHMARK (engine)
# =============================================================================

def run_benchmark_json(
    run_single_question: Callable[..., str],
    loader: GenericDatasetLoader,
    models: List[str],
    num_samples: int,
    show_per_question: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    if loader is None or loader.data is None:
        raise ValueError("Loader não possui dataset carregado")
    if not models:
        raise ValueError("Selecione pelo menos 1 modelo")

    dataset_type = (loader.dataset_type or "").lower().strip() or "open"
    questions = loader.get_subset(num_samples)

    evaluator = BenchmarkEvaluator()
    metrics_selector = MetricsSelector()

    y_true_by_model: Dict[str, List[str]] = {m: [] for m in models}
    y_pred_by_model: Dict[str, List[str]] = {m: [] for m in models}
    open_refs: List[str] = []
    open_preds_by_model: Dict[str, List[str]] = {m: [] for m in models}
    rows: List[Dict[str, Any]] = []

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"📊 BENCHMARK (JSON) — {len(questions)} questões — tipo: {dataset_type.upper()}")
    lines.append("=" * 80)
    lines.append("")

    total_start = time.time()

    for i, q in enumerate(questions, 1):
        qid      = q.get("id", "?")
        question = q.get("question", "")
        expected = q.get("expected_answer", "")

        if not question:
            continue

        if dataset_type == "open":
            open_refs.append(str(expected))

        if show_per_question:
            lines.append("-" * 80)
            lines.append(f"❓ Questão {i}/{len(questions)} — ID: {qid}")
            lines.append("-" * 80)
            lines.append(f"Pergunta: {question}")
            lines.append(f"Esperado: {expected}")
            lines.append("")

        start = time.time()
        full_report = run_single_question(question, use_multi_llm=True, compare_models=models)
        elapsed_ms  = int((time.time() - start) * 1000)

        if show_per_question:
            lines.append(f"⏱️ Tempo: {elapsed_ms} ms")

        row: Dict[str, Any] = {
            "id": qid, "question": question, "expected": expected,
            "time_ms": elapsed_ms, "dataset_type": dataset_type,
        }

        for m in models:
            ans = extract_model_response_from_report(full_report, m)
            preview = (ans or "").replace("\n", " ").strip()
            if len(preview) > 160:
                preview = preview[:160] + "..."
            if show_per_question:
                lines.append(f"📝 {m.upper():8} → {preview}")
            row[m] = ans or ""

            if dataset_type == "closed":
                y_true_by_model[m].append(str(expected).strip().lower())
                y_pred_by_model[m].append(evaluator.extract_answer(ans))
            else:
                open_preds_by_model[m].append(ans or "")

        rows.append(row)
        lines.append("")

    total_ms = int((time.time() - total_start) * 1000)

    lines.append("=" * 80)
    lines.append("🏆 RESUMO")
    lines.append("=" * 80)
    lines.append(f"⏱️ Tempo total: {total_ms} ms")
    lines.append("")

    if dataset_type == "closed":
        lines.append("📊 Dataset CLOSED: métricas completas por modelo")
        lines.append("-" * 80)
        for m in models:
            y_true = y_true_by_model[m]
            y_pred = y_pred_by_model[m]
            if not y_true:
                lines.append(f"{m.upper():10} → ⚠️ sem dados")
                continue
            met  = metrics_selector.calculate_closed_metrics(y_true, y_pred)
            lines.append(f"\n{m.upper()}:")
            lines.append(f"  Accuracy : {met.get('accuracy', 0.0):.4f}")
            lines.append(f"  Precision: {met.get('precision', 0.0):.4f}")
            lines.append(f"  Recall   : {met.get('recall', 0.0):.4f}")
            lines.append(f"  F1       : {met.get('f1', 0.0):.4f}")
            labels = met.get("labels")
            cm     = met.get("confusion_matrix")
            if labels and cm:
                lines.append(f"  Confusion Matrix:")
                lines.append(f"    Labels: {labels}")
                lines.append(f"    Matrix: {cm}")
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines), rows

    lines.append("📝 Dataset OPEN: métricas de texto livre (por modelo)")
    lines.append("-" * 80)
    for m in models:
        preds = open_preds_by_model.get(m, [])
        if not preds:
            lines.append(f"\n{m.upper()}: ⚠️ sem respostas")
            continue
        met = metrics_selector.calculate_open_metrics(open_refs, preds)
        lines.append(f"\n{m.upper()}:")
        if met.get("bleu")               is not None: lines.append(f"  BLEU:                   {met['bleu']:.4f}")
        if met.get("meteor")             is not None: lines.append(f"  METEOR:                 {met['meteor']:.4f}")
        if met.get("bertscore_f1")       is not None: lines.append(f"  BERTScore F1:           {met['bertscore_f1']:.4f}")
        if met.get("semantic_similarity") is not None: lines.append(f"  Similaridade Semântica: {met['semantic_similarity']:.4f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines), rows


# =============================================================================
# 4) WRAPPER COMPATÍVEL COM base.py
# =============================================================================

def run_json_benchmark(*args, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
    run_single_question = kwargs.pop("run_single_question", None)
    respond_fn          = kwargs.pop("respond_fn", None)

    if run_single_question is None and respond_fn is None:
        if args and callable(args[0]):
            run_single_question = args[0]
            args = args[1:]

    if run_single_question is None and respond_fn is not None:
        def _adapt(question: str, use_multi_llm: bool = True, compare_models: Optional[List[str]] = None):
            try:
                return respond_fn(question, use_multi_llm=use_multi_llm, compare_models=compare_models)
            except TypeError:
                return respond_fn(question, use_multi_llm, compare_models)
        run_single_question = _adapt

    if run_single_question is None:
        raise ValueError("run_json_benchmark: não recebeu respond_fn nem run_single_question")

    loader = kwargs.pop("loader", None)
    if loader is None and args:
        loader = args[0]; args = args[1:]
    if loader is None:
        raise ValueError("run_json_benchmark: loader não informado")

    models = kwargs.pop("models", None)
    if models is None and args:
        models = args[0]; args = args[1:]
    if not models:
        raise ValueError("run_json_benchmark: models vazio")

    num_samples   = kwargs.pop("num_samples", None)
    num_questions = kwargs.pop("num_questions", None)
    if num_samples is None and num_questions is not None:
        num_samples = num_questions
    if num_samples is None:
        num_samples = args[0] if args else 3

    show_per_question = kwargs.pop("show_per_question", True)

    return run_benchmark_json(
        run_single_question=run_single_question,
        loader=loader,
        models=models,
        num_samples=int(num_samples),
        show_per_question=bool(show_per_question),
    )