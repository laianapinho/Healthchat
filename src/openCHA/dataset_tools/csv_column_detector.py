"""
CSV Column Detector - Detecta automaticamente as colunas de pergunta/resposta
em qualquer CSV, reaproveitando as keywords do DatasetDetector (usado para JSON).

Uso:
    question_col, answer_col, info = detect_csv_qa_columns(df)

Se não conseguir detectar com confiança suficiente, levanta ValueError com
uma mensagem clara mostrando as colunas disponíveis, para facilitar o debug.
"""
from typing import Tuple, Dict, Any
import pandas as pd

from .dataset_detector import DatasetDetector


def detect_csv_qa_columns(
    df: pd.DataFrame,
    confidence_threshold: float = 0.6,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Detecta as colunas de pergunta e resposta em um DataFrame.

    Args:
        df: DataFrame já carregado (pd.read_csv)
        confidence_threshold: confiança mínima para aceitar a detecção
                               (mais permissivo que o padrão do JSON, pois
                               nomes de coluna costumam ser mais diretos)

    Returns:
        Tuple: (question_col, answer_col, info)
        info contém as confidences e a lista de colunas disponíveis.

    Raises:
        ValueError: se não for possível detectar as duas colunas com
                    confiança mínima. A mensagem já lista as colunas
                    disponíveis do CSV para o usuário conferir.
    """
    detector = DatasetDetector()

    # Reaproveita a lógica de _find_field: ela só olha os NOMES das chaves
    # de um dict, então basta criar um "item" fake com as colunas do df.
    fake_item = {col: "" for col in df.columns}

    question_col, q_conf = detector._find_field(fake_item, DatasetDetector.QUESTION_KEYWORDS)
    answer_col, a_conf = detector._find_field(fake_item, DatasetDetector.ANSWER_KEYWORDS)

    info = {
        "question_col": question_col,
        "question_confidence": q_conf,
        "answer_col": answer_col,
        "answer_confidence": a_conf,
        "all_columns": list(df.columns),
    }

    available = ", ".join(f"'{c}'" for c in df.columns)

    if question_col is None or q_conf < confidence_threshold:
        raise ValueError(
            f"Não foi possível detectar automaticamente a coluna de PERGUNTA. "
            f"Colunas disponíveis no CSV: {available}"
        )

    if answer_col is None or a_conf < confidence_threshold:
        raise ValueError(
            f"Não foi possível detectar automaticamente a coluna de RESPOSTA. "
            f"Colunas disponíveis no CSV: {available}"
        )

    if question_col == answer_col:
        raise ValueError(
            f"A mesma coluna ('{question_col}') foi detectada para pergunta e "
            f"resposta, o que não faz sentido. Colunas disponíveis: {available}"
        )

    return question_col, answer_col, info