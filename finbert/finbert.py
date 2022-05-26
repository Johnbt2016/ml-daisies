from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/pebble_tmp/models/finbert")
model = AutoModelForSequenceClassification.from_pretrained("/pebble_tmp/models/finbert")

qa = pipeline('text-classification',model=model, tokenizer=tokenizer)

def give_sentiment(context):
    '''
    Give the overall financial sentiment for a list of sentences

    Parameters:
    - context (list of strings, or a string) : sentences to process

    Returns:
    - a list of dictionaries, one per sentence, with a 'label' and a 'score'

    '''
    result = qa(context)

    return result

