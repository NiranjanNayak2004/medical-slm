from tokenizer import tokenize

text = """
Fever is a common symptom of many conditions.
A person with fever may have an increased body temperature.
Headache is pain or discomfort in the head.
A person may experience headache with fever.
"""

tokens = tokenize(text)

print("Tokens:")
print(tokens)