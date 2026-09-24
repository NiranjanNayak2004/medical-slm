from tokenizer import tokenize

text = """
Fever is a common symptom of many conditions.
A person with fever may have an increased body temperature.
Headache is pain or discomfort in the head.
A person may experience headache with fever.
"""

tokens = tokenize(text)

# Create vocabulary
vocabulary = {}

for token in tokens:
    if token not in vocabulary:
        vocabulary[token] = len(vocabulary)

# Convert words to IDs
token_ids = []

for token in tokens:
    token_ids.append(vocabulary[token])

print("Tokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

context_size = 3

inputs = []
targets = []

for i in range(len(token_ids) - context_size):
    input_sequence = token_ids[i:i + context_size]
    target = token_ids[i + context_size]

    inputs.append(input_sequence)
    targets.append(target)

print("\nTraining pairs:")

for i in range(len(inputs)):
    print("Input:", inputs[i], "Target:", targets[i])