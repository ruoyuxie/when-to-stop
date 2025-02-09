evaluation_prompt = '''
You are an expert model evaluator specializing in natural language understanding. Your task is to determine if a model's answer is correct by comparing it with the provided gold answers, accounting for valid paraphrasing and alternate expressions of the same answers.

[QUESTION]
{question}
[/QUESTION]

[GOLD_ANSWERS]
{correct_answers}
[/GOLD_ANSWERS]

[MODEL_ANSWER]
{model_answer}
[/MODEL_ANSWER]

Evaluation criteria:
- Answer must convey the same core meaning as gold answers
- Partial matches should be marked incorrect
- Additional correct information beyond gold answers is acceptable
- Empty or off-topic responses are incorrect

Your response should strictly ONLY consist of '[[YES]]' if model answers question correctly, or '[[NO]]' if model answers question incorrectly. Omit any other output.

Your response:'''
