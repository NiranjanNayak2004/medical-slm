correct_answer = 1
prediction = 0.7

loss = (correct_answer - prediction) ** 2

print("Correct answer:", correct_answer)
print("Prediction:", prediction)
print("Loss:", loss)