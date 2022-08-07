from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def compute(text, candidate_labels, multi_label=False):
    if multi_label:
        result = classifier(text, candidate_labels, multi_label=True)
    else:
        result = classifier(text, candidate_labels)

    return result

if __name__ == "__main__":
    res = compute(text="let's go to the moon", candidate_labels=['astronomy', 'travel'], multi_label=False)
    print(res)