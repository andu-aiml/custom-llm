from datasets import load_dataset


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_texts = dataset["train"]["text"]
valid_texts = dataset["validation"]["text"]
test_texts  = dataset["test"]["text"]

#print("Number of training samples:", len(train_texts))

def clean_texts(texts):
    return [t.strip() for t in texts if len(t.strip()) > 0]

train_texts = clean_texts(train_texts)