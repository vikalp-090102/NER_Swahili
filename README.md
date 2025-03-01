# NER_Swahili
Named Entity Recognition for Swahili using Inkuba LM Model


Named Entity Recognition for Swahili using Lelapa Inkuba Language Model
Project Overview
This project focuses on Named Entity Recognition (NER) for the Swahili language using the Lelapa Inkuba Language Model (LM). NER is a crucial Natural Language Processing (NLP) task that involves identifying and classifying entities such as names, locations, organizations, and dates within Swahili text.

Given the complex morphology of Swahili, this project fine-tunes the Lelapa InkubaLM-0.4B model on the Masakhane NER dataset to improve accuracy. The fine-tuned model significantly enhances entity recognition, making it a valuable tool for Swahili NLP applications.

Dataset
The project utilizes the Masakhane NER dataset, specifically the Swahili (swa) subset, sourced from Hugging Face. The dataset is structured into three splits:

Training Set: 2,109 examples
Validation Set: 300 examples
Test Set: 604 examples
Each record consists of:

Tokens: Individual words or subwords
NER Tags: Entity labels (e.g., PERSON, LOCATION, ORGANIZATION)
Installation
To set up the environment, install the required dependencies using:

bash
Copy
Edit
pip install transformers datasets seqeval torch numpy pandas wandb
Hugging Face Authentication
To use the Lelapa Inkuba model, you need a Hugging Face access token. Sign up at Hugging Face and generate a token from your settings under Access Tokens. Then, log in using:

bash
Copy
Edit
huggingface-cli login
Or set the token in your script:

python
Copy
Edit
from huggingface_hub import login
login(token="your_huggingface_access_token")
Weights & Biases (W&B) Authentication
The fine-tuning process requires logging into Weights & Biases (W&B) for experiment tracking. Sign up at Weights & Biases and log in using:

bash
Copy
Edit
wandb login
Model Architecture
The InkubaLM-0.4B model is fine-tuned for token classification with 9 NER categories. The training process includes:

Tokenizer: Hugging Face AutoTokenizer
Model: InkubaLM-0.4B pre-trained model
Training Framework: Hugging Face Trainer API
Evaluation Metrics: Precision, Recall, F1-score (seqeval)
Training Configuration
Key hyperparameters:

Learning Rate: 5e-5
Batch Size: Training (1), Evaluation (2)
Epochs: 4
Evaluation Metric: Overall F1-score
Running the Code
To fine-tune and evaluate the model, run:

python
Copy
Edit
python train.py
To test the model on new Swahili text:

python
Copy
Edit
python infer.py --input "Julius Nyerere alizaliwa Tanzania."
Results
The model’s performance across training epochs:

Epoch	F1 Score
1	0.715
2	0.762
3	0.793
Future Work
Fine-tuning on larger, more diverse datasets
Exploring advanced models like GPT-3 and T5 for improved accuracy
Real-world deployment for live NER applications
