from openCHA.evaluation import ResponseEvaluator, ClinicalContext

evaluator = ResponseEvaluator()

query = "Paciente feminina, 58 anos, com dor no peito e falta de ar há 2 horas, hipertensa, em uso de losartana. O que fazer?"
response = """
Paciente feminina de 58 anos com dor no peito e falta de ar há 2 horas apresenta sinais compatíveis com quadro grave.
O histórico de hipertensão e o uso de losartana devem ser considerados.
A recomendação é procurar atendimento de emergência imediatamente.
"""

context = ClinicalContext(
    age=58,
    sex="feminino",
    symptoms=["dor no peito", "falta de ar"],
    conditions=["hipertensão"],
    medications=["losartana"],
    duration="2 horas"
)

result = evaluator.evaluate(
    query=query,
    response=response,
    clinical_context=context,
    expected_topics=["sintomas", "urgência", "conduta", "histórico"]
)

print(result.model_dump())
