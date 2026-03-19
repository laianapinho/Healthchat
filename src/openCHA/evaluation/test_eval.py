from openCHA.evaluation import ResponseEvaluator, ClinicalContext

evaluator = ResponseEvaluator()

query = "Paciente com dor no peito e falta de ar há 2 horas. O que fazer?"
response = "Dor no peito e falta de ar podem ser sinais de urgência. O ideal é procurar atendimento imediato."

context = ClinicalContext(
    symptoms=["dor no peito", "falta de ar"],
    conditions=["hipertensão"]
)

result = evaluator.evaluate(
    query=query,
    response=response,
    clinical_context=context,
    expected_topics=["sintomas", "urgência", "conduta"]
)

print(result.model_dump())
