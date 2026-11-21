"""
Prepara os CSVs para as abas de BERTScore (portugues ou ingles).

lang="pt" usa as colunas terminadas em _pt (no MedPT: question/answer).
lang="en" usa as colunas originais em ingles (no MedPT: question_en/answer_en).
Reconhece o tipo de CSV pelas colunas:
  - PubMedQA : question + context + long_answer + final_decision
  - LiveQA / MedQuAD : question + answer
        (LiveQA pode ter varias respostas separadas por ' ||| ')
  - Qualquer outro (ex.: MedPT, ja em portugues, so com lang="pt"):
        detecta as colunas de pergunta/resposta como sempre foi feito.
  - So pergunta, sem coluna de resposta: mode="no_reference"
        (so gera as respostas dos modelos; nao ha como calcular BERTScore).
"""
import re
from typing import Any, Dict, List

from .csv_column_detector import detect_csv_qa_columns

SEP = "|||"

PROMPTS_PUBMEDQA = {
    "pt": (
        "Contexto: {ctx}\n\nPergunta: {q}\n\n"
        "Responda EXATAMENTE em duas linhas, neste formato, sem nenhum outro texto:\n"
        "Decisão: <sim|não|talvez>\n"
        "Explicação: <uma ou duas frases baseadas no contexto>\n\n"
        "Regras: a primeira linha deve começar com 'Decisão:' seguida de APENAS uma "
        "das palavras sim, não ou talvez. Não use negrito, listas nem markdown. "
        "Não escreva nada antes de 'Decisão:'.\n\n"
        "Exemplo:\n"
        "Decisão: não\n"
        "Explicação: O estudo não encontrou melhora significativa."
    ),
    "en": (
        "Context: {ctx}\n\nQuestion: {q}\n\n"
        "You MUST answer in exactly two lines, in this format, with no other text:\n"
        "Decision: <yes|no|maybe>\n"
        "Explanation: <one or two sentences based on the context>\n\n"
        "Rules: the first line must start with 'Decision:' followed by ONLY one of "
        "the words yes, no or maybe. Do not use bold, bullet points or markdown. "
        "Do not write anything before 'Decision:'.\n\n"
        "Example:\n"
        "Decision: no\n"
        "Explanation: The study found no significant improvement."
    ),
}

_DEC_MAP = {
    "sim": "yes", "yes": "yes",
    "não": "no", "nao": "no", "no": "no",
    "talvez": "maybe", "maybe": "maybe",
}

# ------------------------------------------------------------------
# Auxiliares
# ------------------------------------------------------------------
def _col(df, name):
    """Acha a coluna ignorando maiusculas/minusculas."""
    return {c.lower(): c for c in df.columns}.get(name.lower())


def _clean_text(text) -> str:
    """Tira espacos invisiveis do Google Tradutor, quebras de linha e espacos repetidos."""
    text = str(text).replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def _drop_empty(df, cols):
    """Remove linhas em que alguma das colunas esta vazia."""
    mask = True
    for c in cols:
        mask = mask & df[c].notna() & (df[c].astype(str).str.strip() != "")
    return df[mask].reset_index(drop=True)


def _check_formulas(df, cols):
    for c in cols:
        if df[c].astype(str).str.startswith("=").any():
            raise ValueError(
                f"A coluna '{c}' ainda tem fórmulas do Sheets (começam com '='). "
                "Cole como 'somente valores' e baixe o CSV de novo."
            )


def _sample(df, n, use_random):
    n = min(int(n), len(df))
    if use_random:
        return df.sample(n=n, random_state=42).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


# ------------------------------------------------------------------
# Preparacao principal
# ------------------------------------------------------------------
def prepare_pt_items(df, n_samples, use_random, lang="pt") -> Dict[str, Any]:
    if lang not in PROMPTS_PUBMEDQA:
        raise ValueError(f"Idioma '{lang}' não suportado (use 'pt' ou 'en').")

    # O MedPT original é em português (colunas question/answer) e, quando
    # traduzido, ganha question_en/answer_en. Os outros datasets usam
    # question_pt (português) e question (inglês original).
    is_medpt = bool(_col(df, "question_type") and _col(df, "medical_specialty"))

    def pick(name):
        if lang == "pt":
            return _col(df, name + "_pt") or (_col(df, name) if is_medpt else None)
        return _col(df, name + "_en") or (None if is_medpt else _col(df, name))

    q_pt, a_pt = pick("question"), pick("answer")
    ctx_pt, la_pt = pick("context"), pick("long_answer")
    fd = _col(df, "final_decision")

    # MedPT em inglês só funciona com as colunas traduzidas
    if lang == "en" and is_medpt and not (q_pt and a_pt):
        raise ValueError(
            "O MedPT está em português. Para avaliar em inglês, o CSV precisa "
            "das colunas 'question_en' e 'answer_en'."
        )

    # ── PubMedQA ──
    if q_pt and ctx_pt and la_pt and fd:
        cols = [q_pt, ctx_pt, la_pt, fd]
        _check_formulas(df, cols)
        df = _sample(_drop_empty(df, cols), n_samples, use_random)
        return {
            "mode": "pubmedqa",
            "prompts": [
                PROMPTS_PUBMEDQA[lang].format(ctx=_clean_text(c), q=_clean_text(q))
                for c, q in zip(df[ctx_pt], df[q_pt])
            ],
            "questions": [_clean_text(q) for q in df[q_pt]],
            "references": [[_clean_text(r)] for r in df[la_pt]],
            "labels": df[fd].astype(str).str.strip().str.lower().tolist(),
        }

    # ── LiveQA / MedQuAD / MedPT ──
    if q_pt and a_pt:
        cols = [q_pt, a_pt]
        _check_formulas(df, cols)
        df = _sample(_drop_empty(df, cols), n_samples, use_random)
        questions = [_clean_text(q) for q in df[q_pt]]
        references = []
        for a in df[a_pt]:
            parts = [_clean_text(p) for p in str(a).split(SEP)]
            references.append([p for p in parts if p])
        return {
            "mode": "qa",
            "prompts": questions,
            "questions": questions,
            "references": references,
            "labels": None,
        }

    # ── Outros CSVs: comportamento de sempre ──
    try:
        q, a, _ = detect_csv_qa_columns(df)
    except ValueError:
        # Sem coluna de resposta: se ao menos a pergunta existe, só gera as
        # respostas dos modelos (sem referência, não há BERTScore).
        if not q_pt:
            raise
        _check_formulas(df, [q_pt])
        df = _sample(_drop_empty(df, [q_pt]), n_samples, use_random)
        questions = [_clean_text(x) for x in df[q_pt]]
        return {
            "mode": "no_reference",
            "prompts": questions,
            "questions": questions,
            "references": [[] for _ in questions],
            "labels": None,
        }

    df = _sample(_drop_empty(df, [q, a]), n_samples, use_random)
    questions = df[q].astype(str).tolist()
    return {
        "mode": "default",
        "prompts": questions,
        "questions": questions,
        "references": [[str(r)] for r in df[a]],
        "labels": None,
    }


# ------------------------------------------------------------------
# Veredito sim/nao/talvez (PubMedQA)
# ------------------------------------------------------------------
def extract_decision(text):
    text = text or ""
    m = re.search(
        r"(?:decis[ãa]o|decision)\W{0,6}(sim|n[ãa]o|talvez|yes|no|maybe)\b",
        text, re.I)
    if not m:  # fallback: resposta que já começa com a palavra
        m = re.match(r"\W*(sim|n[ãa]o|talvez|yes|no|maybe)\b", text, re.I)
    return _DEC_MAP.get(m.group(1).lower()) if m else None

def extract_explanation(text):
    m = re.search(r"(?:explica[çc][ãa]o|explanation)\s*:\s*(.*)",
                  text or "", re.I | re.S)
    return m.group(1).strip() if m else (text or "")


def decision_metrics(records: List[Dict[str, Any]], models: List[str]) -> str:
    labels = ["yes", "no", "maybe"]
    lines = ["### 🎯 Acurácia do veredito (sim/não/talvez)\n"]
    for m in models:
        rs = [r for r in records if r["modelo"] == m and "final_decision" in r]
        if not rs:
            continue
        y_true = [r["final_decision"] for r in rs]
        y_pred = [r["predicted"] or "invalid" for r in rs]
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(rs)

        f1s = []
        for lab in labels:
            tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
            fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
            fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

        invalid = y_pred.count("invalid")
        lines.append(
            f"- **{m.upper()}** → Acc={acc:.4f} | F1 macro={sum(f1s)/3:.4f} "
            f"| sem veredito válido: {invalid}/{len(rs)}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Nota com varias referencias: fica a melhor
# ------------------------------------------------------------------
def best_record(evaluator, pergunta, modelo, resposta, references):
    """Compara a resposta com cada referencia e guarda a de maior F1."""
    best = None
    for ref in references:
        rec = evaluator.build_record(
            pergunta=pergunta, modelo=modelo,
            resposta=resposta, reference=ref)
        if best is None or rec["F1"] > best["F1"]:
            best = rec
    best["n_referencias"] = len(references)
    return best