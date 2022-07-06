# FinBERT: Financial Sentiment Analysis with BERT

Daisi created from the FinBERT sentiment analysis model available on [Hugging Face model hub](https://huggingface.co/ProsusAI/finbert).

FinBERT is a pre-trained NLP model to analyze sentiment of financial text.
It is built by further training the BERT language model in the finance domain,
using a large financial corpus and thereby fine-tuning it for financial sentiment classification.
For the details, please see FinBERT: [Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/pdf/1908.10063.pdf)

How to use :

```python
import pydaisi as pyd

finbert = pyd.Daisi("laiglejm/Finbert")
context = 'Apple (AAPL) Stock Price: Why It Increased Over 1.9% Today'
answer = finbert.give_sentiment(context).value

```
