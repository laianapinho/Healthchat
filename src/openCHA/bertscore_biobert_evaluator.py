# openCHA/bertscore_biobert_evaluator.py
"""
Módulo isolado para cálculo e persistência do BERTScore usando o BioBERT
como modelo de embeddings (dmis-lab/biobert-base-cased-v1.2).

Usado pela aba 4 do base.py (Benchmark CSV + BERTScore) quando se deseja
avaliação com um modelo biomédico especializado, em vez do BERT genérico.

Observação: o BioBERT foi treinado em textos biomédicos em INGLÊS
(PubMed/PMC). Use este avaliador preferencialmente com o conjunto
MedDataProcessed (inglês). Para o MedPT (português), o BioBERT não é
recomendado, pois não foi treinado nesse idioma.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from bert_score import BERTScorer

logger = logging.getLogger(__name__)

# Configuração fixa do modelo BioBERT
BIOBERT_MODEL_TYPE = "dmis-lab/biobert-base-cased-v1.2"
BIOBERT_NUM_LAYERS = 12   # camada usada para extrair os embeddings
BIOBERT_MAX_LENGTH = 512  # limite real de tokens do BERT


class BioBertScoreEvaluator:
    """
    Encapsula cálculo de BERTScore com BioBERT e salvamento de resultados.
    """

    def __init__(self, batch_size: int = 16, verbose: bool = True):
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_type = BIOBERT_MODEL_TYPE
        self.num_layers = BIOBERT_NUM_LAYERS

        # O modelo é carregado UMA vez aqui (antes era recarregado a cada chamada)
        self.scorer = BERTScorer(
            model_type=self.model_type,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
        )
        # CORREÇÃO do erro "int too big to convert":
        # o tokenizer do BioBERT vem com model_max_length gigante.
        self.scorer._tokenizer.model_max_length = BIOBERT_MAX_LENGTH

    # ------------------------------------------------------------------
    # Cálculo
    # ------------------------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calcula BERTScore (com BioBERT) para listas de hipóteses e referências.
        Retorna (P, R, F1) como listas de floats.
        """
        P, R, F1 = self.scorer.score(
            hypotheses,
            references,
            verbose=self.verbose,  # mostra a barra de progresso no terminal
        )
        return P.tolist(), R.tolist(), F1.tolist()

    def compute_single(
        self,
        hypothesis: str,
        reference: str,
    ) -> Tuple[float, float, float]:
        """
        Calcula BERTScore (BioBERT) para um único par hipótese/referência.
        Retorna (P, R, F1) como floats.
        """
        P, R, F1 = self.compute([hypothesis], [reference])
        return P[0], R[0], F1[0]

    # ------------------------------------------------------------------
    # Construção de registros
    # ------------------------------------------------------------------

    def build_record(
        self,
        pergunta: str,
        modelo: str,
        resposta: str,
        reference: str,
    ) -> Dict[str, Any]:
        """
        Calcula BERTScore (BioBERT) e retorna um dict pronto para salvar/exibir.
        Retorna P=R=F1=0 se a resposta estiver vazia ou ocorrer erro.
        """
        # ── guarda contra resposta vazia (ex: modelo que falhou) ──────
        if not resposta or not resposta.strip():
            logger.warning(f"BioBERTScore [{modelo}]: resposta vazia — pulando cálculo")
            print(f"⚠️ BioBERTScore [{modelo}]: resposta vazia — pulando cálculo")
            p_val = r_val = f1_val = 0.0
        else:
            print(f"🧮 Calculando BioBERTScore [{modelo}]...")
            try:
                p_val, r_val, f1_val = self.compute_single(resposta, reference)
                print(
                    f"✅ BioBERTScore [{modelo}]: "
                    f"P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f}"
                )
            except Exception as e:
                logger.error(f"Erro BioBERTScore [{modelo}]: {e}")
                print(f"❌ Erro BioBERTScore [{modelo}]: {e}")
                p_val = r_val = f1_val = 0.0
        # ─────────────────────────────────────────────────────────────

        return {
            "timestamp": datetime.now().isoformat(),
            "pergunta":  pergunta,
            "modelo":    modelo,
            "resposta":  resposta,
            "P":         round(p_val,  4),
            "R":         round(r_val,  4),
            "F1":        round(f1_val, 4),
        }

    # ------------------------------------------------------------------
    # Resumo
    # ------------------------------------------------------------------

    def summarize(
        self,
        records: List[Dict[str, Any]],
        models: List[str],
        n: int,
    ) -> str:
        """
        Gera texto de resumo com médias de P/R/F1 por modelo (BioBERT).
        Modelos com resposta vazia (F1=0) são marcados com ⚠️.
        """
        lines = [f"### 📈 Médias por modelo — BioBERTScore ({n} questão/ões)\n"]
        for model in models:
            mrs = [r for r in records if r["modelo"] == model]
            if not mrs:
                continue
            # separa registros com e sem resposta
            ok  = [r for r in mrs if r["resposta"].strip()]
            nok = [r for r in mrs if not r["resposta"].strip()]

            if ok:
                avg_p  = sum(r["P"]  for r in ok) / len(ok)
                avg_r  = sum(r["R"]  for r in ok) / len(ok)
                avg_f1 = sum(r["F1"] for r in ok) / len(ok)
                nota   = f" ⚠️ {len(nok)} sem resposta" if nok else ""
                lines.append(
                    f"- **{model.upper()}** → "
                    f"P={avg_p:.4f} | R={avg_r:.4f} | F1={avg_f1:.4f}{nota}"
                )
            else:
                lines.append(f"- **{model.upper()}** → ⚠️ todas as respostas falharam")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        records: List[Dict[str, Any]],
        file_path: str = "results_multillm_biobert.json",
    ) -> str:
        """
        Salva registros em JSON acumulativo (append).
        Retorna o caminho do arquivo salvo.
        """
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        else:
            old_data = []

        old_data.extend(records)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, indent=2, ensure_ascii=False)

        logger.info(f"BioBERTScore: {len(records)} registros salvos em '{file_path}'")
        print(f"💾 BioBERTScore: {len(records)} registros salvos em '{file_path}'")
        return file_path