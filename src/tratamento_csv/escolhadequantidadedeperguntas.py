import pandas as pd

file_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_perguntas_respostas_avaliacao4mais.csv"
output_path = "/home/laiana/Documentos/Healthchat/src/openCHA/datasets/corpus_amostra_120.csv"

MIN_PERGUNTAS_ESPECIALIDADE = 8   # exclui especialidades com menos que isso
TOTAL_AMOSTRA = 120               # tamanho final desejado
MIN_POR_ESPECIALIDADE = 2         # piso: toda especialidade incluída aparece pelo menos 2x
TETO_POR_ESPECIALIDADE = 15       # teto: nenhuma especialidade passa disso
SEED = 42                         # fixo, pra amostra ser sempre a mesma se rodar de novo

df = pd.read_csv(file_path)

print(f"Total de linhas: {len(df)}")
print(f"Total de especialidades: {df['Especialidade'].nunique()}")
print()

# ── 1) corta especialidades com poucas perguntas ──────────────────
contagem = df['Especialidade'].value_counts()
especialidades_validas = contagem[contagem >= MIN_PERGUNTAS_ESPECIALIDADE].index
df_valido = df[df['Especialidade'].isin(especialidades_validas)].copy()

print(f"Especialidades excluídas (< {MIN_PERGUNTAS_ESPECIALIDADE} perguntas): "
      f"{contagem[contagem < MIN_PERGUNTAS_ESPECIALIDADE].shape[0]}")
print(f"Especialidades usadas na amostragem: {len(especialidades_validas)}")
print()

# ── 2) calcula quantas perguntas tirar de cada especialidade ──────
contagem_valida = df_valido['Especialidade'].value_counts()
proporcao = contagem_valida / contagem_valida.sum()

n_por_especialidade = (proporcao * TOTAL_AMOSTRA).round().astype(int)
n_por_especialidade = n_por_especialidade.clip(lower=MIN_POR_ESPECIALIDADE, upper=TETO_POR_ESPECIALIDADE)
n_por_especialidade = n_por_especialidade.combine(contagem_valida, min)  # não pega mais do que existe

# ── 3) seleciona de fato as perguntas, especialidade por especialidade ──
partes = []
for especialidade, n in n_por_especialidade.items():
    grupo = df_valido[df_valido['Especialidade'] == especialidade]
    selecionado = grupo.sample(n=n, random_state=SEED)
    partes.append(selecionado)

df_amostra = pd.concat(partes).sort_values(by='ID').reset_index(drop=True)

# ── 4) mostra quanto ficou de cada especialidade ───────────────────
print("Quantidade final por especialidade na amostra:")
print(df_amostra['Especialidade'].value_counts().sort_values(ascending=False))
print()
print(f"Total da amostra: {len(df_amostra)} (alvo era {TOTAL_AMOSTRA})")

# ── 5) salva o CSV ──────────────────────────────────────────────
df_amostra.to_csv(output_path, index=False)
print(f"\nArquivo salvo em: {output_path}")