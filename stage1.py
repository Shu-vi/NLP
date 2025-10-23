import json
import spacy

nlp = spacy.load("ru_core_news_sm")

input_path = "corpus.jsonl"
output_path = "processed_corpus.jsonl"

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(lemmas)

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            data["text"] = preprocess_text(data["text"])
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")

print("Готово")
