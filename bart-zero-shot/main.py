from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def compute(text, candidate_labels, multi_class=False):
    candidate_labels = ['travel', 'cooking', 'dancing']
    result = classifier(text, candidate_labels, multi_class)

    return result

