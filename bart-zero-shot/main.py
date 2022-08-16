from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def compute_class(text, candidate_labels, multi_label=False):
    candidate_labels = [c.strip() for c in candidate_labels.split(',')]
    if multi_label:
        result = classifier(text, candidate_labels, multi_label=True)
    else:
        result = classifier(text, candidate_labels)

    return result

if __name__ == "__main__":
    res = compute_class(text="let's go to the moon", candidate_labels='astronomy, travel', multi_label=False)
    print(res)