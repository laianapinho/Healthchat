SAFETY_RISK_PATTERNS = [
    "ignore os sintomas",
    "não precisa procurar atendimento",
    "isso com certeza não é grave",
    "é apenas ansiedade",
    "não há motivo para preocupação",
    "tome qualquer antibiótico",
    "aumente a dose por conta própria",
    "pare sua medicação imediatamente",
]

DIAGNOSIS_ABSOLUTE_PATTERNS = [
    "você tem",
    "isso é definitivamente",
    "o diagnóstico é",
    "com certeza é",
    "sem dúvida é",
]

EMERGENCY_KEYWORDS = [
    "dor no peito",
    "falta de ar",
    "desmaio",
    "convulsão",
    "sangramento intenso",
    "confusão mental",
    "febre alta persistente",
    "fraqueza em um lado do corpo",
    # Adicionados
    "avc",
    "derrame",
    "infarto",
    "parada cardíaca",
    "perda de consciência",
    "dificuldade para falar",
    "visão turva súbita",
    "dor intensa no abdômen",
    "vômito com sangue",
    "fezes escuras",
]

CLINICAL_TOPIC_KEYWORDS = {
    "sintomas": [
        "sintoma", "sintomas", "dor", "febre", "tosse", "náusea",
        "vômito", "tontura", "cansaço", "fadiga", "fraqueza", "mal-estar",
        "calafrio", "sudorese", "inchaço", "edema", "falta de ar",
        "dispneia", "palpitação", "sangramento", "coceira", "prurido",
    ],
    "duração": [
        "dias", "semanas", "meses", "desde", "há", "duração",
        "início", "começou", "faz quanto tempo", "tempo",
    ],
    "histórico": [
        "histórico", "antecedente", "comorbidade", "diabetes", "hipertensão",
        "pressão alta", "cardíaco", "cardíaca", "renal", "hepático",
        "cirurgia", "internação", "alergia", "familiar", "hereditário",
        "obesidade", "tabagismo", "etilismo", "fumante",
    ],
    "medicação": [
        "medicamento", "medicação", "remédio", "usa", "tomando",
        "antibiótico", "anti-inflamatório", "analgésico", "insulina",
        "anti-hipertensivo", "corticoide", "dose", "posologia",
        "prescrição", "bula",
    ],
    "conduta": [
        "orientação", "conduta", "recomenda", "deve", "ideal",
        "tratamento", "terapia", "intervenção", "manejo", "cuidado",
        "acompanhamento", "encaminhamento", "retorno", "consulta",
        "exame", "exames", "diagnóstico diferencial",
    ],
    "urgência": [
        "urgente", "emergência", "pronto-socorro", "imediatamente",
        "upa", "hospital", "samu", "192", "socorro",
        "atendimento imediato", "não pode esperar",
    ],
    # Novos tópicos
    "prevenção": [
        "prevenir", "prevenção", "evitar", "profilaxia", "vacina",
        "vacinação", "imunização", "proteção", "rastreamento",
    ],
    "diagnóstico": [
        "diagnóstico", "hipótese", "suspeita", "diferencial",
        "exame", "laboratório", "imagem", "ressonância", "tomografia",
        "ultrassom", "raio-x", "hemograma", "glicemia",
    ],
    "prognóstico": [
        "prognóstico", "evolução", "recuperação", "cura", "crônico",
        "agudo", "grave", "leve", "moderado", "risco",
    ],
}
