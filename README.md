# NLP Project: Swahili Named Entity Recognition using InkubaLM-0.4B

## Project Overview
This project fine-tunes the **Lelapa/InkubaLM-0.4B** language model for Named Entity Recognition (NER) on the **Swahili News dataset**. The goal is to improve entity recognition capabilities in Swahili by leveraging **transformers** and **Hugging Face's datasets**.

## Prerequisites
Before running the project, ensure you have **Python 3.13.0** installed on your system.

### Required Libraries
Install the necessary dependencies by running:
```bash
pip install transformers datasets seqeval torch
```

Additionally, you need to authenticate with **Hugging Face**:
```bash
huggingface-cli login
```

## Dataset
We use the **MasakhaNER** dataset for Swahili NER, which is loaded via Hugging Face's `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("masakhaner", "swa")
print(dataset)
```

## Model and Tokenization
We fine-tune **Lelapa/InkubaLM-0.4B** using the `AutoTokenizer` and `AutoModelForTokenClassification`:
```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

## Training
We use Hugging Face's `Trainer` for fine-tuning. The `DataCollatorForTokenClassification` ensures proper token alignment.
```python
from transformers import AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from datasets import Dataset

model = AutoModelForTokenClassification.from_pretrained("lelapa/InkubaLM-0.4B", num_labels=NUM_LABELS)
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)
trainer.train()
```

## Evaluation
We evaluate the model using the **seqeval** metric:
```python
from seqeval.metrics import classification_report
predictions = trainer.predict(dataset["test"])
y_pred = predictions.predictions.argmax(-1)
y_true = dataset["test"]["labels"]
print(classification_report(y_true, y_pred))
```

## Usage
After training, you can use the model to predict named entities in Swahili text.
```python
text = "Rais wa Kenya William Ruto alihutubia taifa."
tokens = tokenizer(text, return_tensors="pt")
predictions = model(**tokens)
print(predictions)
```

## Future Work
- Experiment with different pre-trained models.
- Improve entity recognition by using additional Swahili corpora.
- Deploy the model as an API for real-time NER tasks.

## Author
Vikalp Srivastava

## License
This project is for research purposes only.

