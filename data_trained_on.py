from datasets import load_dataset

dataset = load_dataset("squad")
small_train = dataset["train"].select(range(1000))

for i in range(5):
    question = small_train[i]["question"]
    context = small_train[i]["context"]
    answers = small_train[i]["answers"]["text"]  
    print(f"\nExample {i+1}")
    print(f"Q: {question}")
    print(f"A: {answers}")
