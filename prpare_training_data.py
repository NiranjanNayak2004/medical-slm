from tokenizer import tokenize

# Read medical training text
with open("data/medical_text.txt", "r") as file:
    text = file.read()

# Convert text into tokens
tokens = tokenize(text)

# Create vocabulary
vocabulary = {}

for token in tokens:
    if token not in vocabulary:
        vocabulary[token] = len(vocabulary)

# Convert tokens to IDs
token_ids = []

for token in tokens:
    token_ids.append(vocabulary[token])

# Create training sequences
context_size = 3

X = []
y = []

for i in range(len(token_ids) - context_size):

    input_sequence = token_ids[i:i + context_size]

    target = token_ids[i + context_size]

    X.append(input_sequence)
    y.append(target)

print("Number of tokens:", len(tokens))
print("Vocabulary size:", len(vocabulary))

print("\nFirst 5 training examples:")

for i in range(5):
    print(
        "Input:",
        X[i],
        "Target:",
        y[i]
    )