"""
Lê o documento de rubrica (.pdf ou .docx) e devolve o texto de cada critério,
para ser inserido em {rubrica_criterio} do PROMPT_AVALIACAO.

POR QUE ESTE ARQUIVO EXISTE
    O documento traz os 5 critérios juntos, mas o LLM Judge avalia UM critério
    por chamada. Este módulo recorta o documento e entrega, para cada chamada,
    só a seção do critério que está sendo avaliado.

O texto vem DIRETO do documento (seções 3.1 a 3.5 da Parte II): título,
definição, exemplo e tabela de notas 1-5. Nada é digitado à mão.

RESULTADO FINAL (para qualquer formato):
    {"corretude": "<texto da seção 3.1>", "completude": "<3.2>", ...}
"""
from __future__ import annotations

import glob
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict

from docx import Document                     # leitura de .docx
from docx.table import Table                  # representa uma tabela do Word
from docx.text.paragraph import Paragraph     # representa um parágrafo do Word
from pypdf import PdfReader                   # leitura de .pdf

# Documento usado quando nenhum arquivo é enviado pela interface:
# procura, na mesma pasta deste arquivo .py, qualquer arquivo cujo nome comece
# com "Protocolo_e_Rubrica" (ex.: "..._1.docx"). O .docx tem prioridade sobre o
# .pdf, porque a leitura do .docx é mais confiável.
_PASTA = os.path.dirname(os.path.abspath(__file__))
_PREFIXO_PADRAO = "Protocolo_e_Rubrica"


def _achar_padrao() -> str:
    for ext in (".docx", ".pdf"):
        achados = sorted(glob.glob(os.path.join(_PASTA, _PREFIXO_PADRAO + "*" + ext)))
        if achados:
            return achados[0]
    # Nenhum encontrado: devolve um caminho-exemplo; o erro "não encontrado"
    # mostrará este caminho, para você saber onde colocar o arquivo.
    return os.path.join(_PASTA, _PREFIXO_PADRAO + ".docx")


DOCX_PADRAO = _achar_padrao()   # (nome mantido por compatibilidade com o llmjudge.py)

# Liga o número da seção no documento à chave do critério usada no llmjudge.py.
# Ex.: a seção "3.3. Segurança..." vira o critério "seguranca".
SECOES = {
    "3.1": "corretude",
    "3.2": "completude",
    "3.3": "seguranca",
    "3.4": "clareza",
    "3.5": "relevancia",
}


def _blocos_em_ordem(doc):
    """
    Percorre o corpo do .docx na ordem em que as coisas aparecem na página.

    O python-docx separa "parágrafos" e "tabelas" em listas diferentes, o que
    perde a ordem (a tabela de notas ficaria longe do texto dela). Por isso
    lemos o XML do corpo diretamente, na sequência original.
    """
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):          # <w:p>  = parágrafo
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):      # <w:tbl> = tabela
            yield Table(child, doc)


def _normalizar(texto: str) -> str:
    """Minúsculas e sem acentos ("Segurança" -> "seguranca"), para comparar sem se
    preocupar com acentuação ou maiúsculas."""
    sem_acento = unicodedata.normalize("NFD", texto)
    sem_acento = "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")
    return sem_acento.lower().strip()


# Título de um critério: número opcional ("3.1." / "3.1" / "3 . 1") + NOME + travessão.
# Ex.: "3.3. Segurança — A resposta pode representar riscos...?"
# Procurar pelo NOME (e não só pelo número) funciona mesmo que o PDF tenha
# perdido ou alterado a numeração. Em "Corretude: verifica..." (seção 1 da
# rubrica) o nome vem seguido de ":" e não de travessão, então não confunde.
_TITULO_CRITERIO = re.compile(
    r"^(?:3\s*\.\s*[1-5]\s*\.?\s*)?(corretude|completude|seguranca|clareza|relevancia)\s*[—–-]\s")

# Linhas que indicam o FIM da seção dos critérios (início de outra seção).
_FIM_SECOES = re.compile(
    r"^(\d\.\s|orientacoes gerais|registro das avaliacoes|consideracoes finais)")


def _texto_do_pdf(caminho: str) -> str:
    """Extrai o texto do PDF com pypdf; se vier vazio, tenta pdfplumber."""
    texto = "\n".join((p.extract_text() or "") for p in PdfReader(caminho).pages)
    if len(texto.strip()) < 50:
        try:
            import pdfplumber
            with pdfplumber.open(caminho) as pdf:
                texto = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            pass
    return texto


def _carregar_pdf(caminho: str) -> Dict[str, str]:
    """PDF: extrai o texto e recorta os 5 critérios pelo NOME do título."""
    texto = _texto_do_pdf(caminho)

    # Quebra em linhas, tira espaços das pontas e remove linhas vazias e o
    # rodapé de página ("Protocolo experimental e rubrica • 7").
    linhas = [l.strip() for l in texto.splitlines()]
    linhas = [l for l in linhas
              if l and not _normalizar(l).startswith("protocolo experimental e rubrica")]

    rubricas: Dict[str, list] = {}   # critério -> lista de linhas do texto dele
    atual = None                     # critério que estamos lendo agora (None = nenhum)

    for l in linhas:
        norm = _normalizar(l)

        # Início de um critério?
        m = _TITULO_CRITERIO.match(norm)
        if m:
            atual = m.group(1)                    # já é a chave: "seguranca", "clareza"...
            rubricas[atual] = [f"CRITÉRIO: {l}"]  # o título abre o bloco
            continue

        # Início de outra seção (ex.: "4. Orientações gerais") = fim dos critérios.
        if _FIM_SECOES.match(norm):
            atual = None
            continue

        # Linha comum: pertence ao critério que estamos lendo.
        if atual:
            rubricas[atual].append(l)

    # Se faltar critério, o arquivo não é o documento certo, ou é um PDF
    # digitalizado (imagem, sem texto). Mostra o que foi lido para facilitar o diagnóstico.
    faltando = set(SECOES.values()) - set(rubricas)
    if faltando:
        previa = " ".join(texto.split())[:200] or "(NENHUM texto extraído — PDF provavelmente digitalizado)"
        raise ValueError(
            f"Critérios não encontrados no PDF: {sorted(faltando)}. "
            f"Caracteres extraídos: {len(texto)}. Início do texto lido: {previa!r}")

    # Une as linhas de cada critério em um único texto.
    return {k: "\n".join(v) for k, v in rubricas.items()}


@lru_cache(maxsize=4)   # guarda o resultado: o arquivo é lido uma vez só, não a cada chamada
def carregar_rubricas(caminho_docx: str = DOCX_PADRAO) -> Dict[str, str]:
    """
    Lê o arquivo (PDF ou DOCX) e devolve {critério: texto da rubrica}.
    Levanta FileNotFoundError / ValueError se o arquivo não servir.
    """
    if not os.path.exists(caminho_docx):
        raise FileNotFoundError(
            f"Documento de rubrica não encontrado: {caminho_docx}")

    # PDF segue o caminho próprio; o resto é tratado como .docx abaixo.
    if caminho_docx.lower().endswith(".pdf"):
        return _carregar_pdf(caminho_docx)

    doc = Document(caminho_docx)
    rubricas: Dict[str, list] = {}   # critério -> lista de blocos de texto
    atual = None                     # critério que estamos lendo agora

    for bloco in _blocos_em_ordem(doc):

        # ---- Parágrafos -----------------------------------------------
        if isinstance(bloco, Paragraph):
            texto = bloco.text.strip()
            if not texto:                     # ignora parágrafos vazios
                continue

            # Título de subseção, no formato "3.3. Segurança — ..." (também
            # pega "4.1", "2.1" etc.; o SECOES.get filtra só 3.1 a 3.5).
            m = re.match(r"^(\d\.\d)\.?\s", texto)
            if m:
                atual = SECOES.get(m.group(1))   # None se não for 3.1–3.5
                if atual:
                    rubricas[atual] = [f"CRITÉRIO: {texto}"]
                continue

            # Título de seção principal, ex.: "4. Orientações gerais".
            # Encerra o critério atual para não misturar com a seção seguinte.
            if re.match(r"^\d\.\s", texto):
                atual = None
                continue

            # Parágrafo comum (definição, exemplo...): pertence ao critério atual.
            if atual:
                rubricas[atual].append(texto)

        # ---- Tabelas --------------------------------------------------
        # A tabela de notas 1-5 de cada critério. Só guardamos se estivermos
        # dentro de um critério (as outras tabelas do documento são ignoradas).
        elif atual:
            linhas = ["Descritores (use exclusivamente estes para atribuir a nota):"]
            for row in bloco.rows[1:]:        # [1:] pula a linha de cabeçalho
                # As 2 primeiras colunas: número da nota e a descrição.
                nota, desc = (c.text.strip() for c in row.cells[:2])
                linhas.append(f"Nota {nota}: {desc}")
            rubricas[atual].append("\n".join(linhas))

    # Confere se os 5 critérios foram encontrados.
    faltando = set(SECOES.values()) - set(rubricas)
    if faltando:
        raise ValueError(f"Critérios não encontrados no documento: {faltando}")

    # Une os blocos de cada critério separando por linha em branco.
    return {k: "\n\n".join(v) for k, v in rubricas.items()}


def rubrica_do_criterio(criterio_key: str, caminho_docx: str = DOCX_PADRAO) -> str:
    """
    Devolve o texto da rubrica de UM critério (é o que entra em {rubrica_criterio}).
    Ex.: rubrica_do_criterio("seguranca", "meu_arquivo.pdf")
    """
    return carregar_rubricas(caminho_docx)[criterio_key]