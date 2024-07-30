

pip install transformers[torch] tokenizers datasets evaluate rouge_score sentencepiece huggingface_hub pandas --upgrade

# Import the necessary libraries
import pandas as pd
import nltk
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
# Load the JSONL file into a DataFrame
df = pd.read_json("hf://datasets/toughdata/quora-question-answer-dataset/Quora-QuAD.jsonl", lines=True)

# Convert the DataFrame to a CSV file
df.to_csv("Quora-QuAD.csv", index=False)

df = pd.read_csv("/content/Quora-QuAD.csv")
df.head()

# Remove duplicates
df = df.drop_duplicates(subset=["question", "answer"])# Check duplicate questions and remove them

print(f"Number of duplicate questions: {len(df) - df['question'].nunique()}")

df.drop_duplicates(subset=['question'],inplace=True)
print("\nNumber of records after removing duplicates: ",len(df))

print(df)
# Convert back to Dataset object
dataset = Dataset.from_pandas(df)

# Split the dataset
dataset = dataset.train_test_split(test_size=0.2)

# Print dataset shape
print(dataset)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# We prefix our tasks with "answer the question"
prefix = "answer the question: "

# Define our preprocessing function
def preprocess_function(examples):
    """Add prefix to the sentences, clean text, tokenize, and set the labels"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        # Remove special characters and digits
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stop words and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    # Clean and preprocess the inputs and labels
    inputs = [prefix + clean_text(doc) for doc in examples["question"]]
    answers = [clean_text(doc) for doc in examples["answer"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=answers, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up evaluation metrics
nltk.download("punkt", quiet=True)
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute metrics
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    f1_result = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, average='weighted')

    # Combine metrics
    result = {
        'rouge1': rouge_result['rouge1'],
        'rouge2': rouge_result['rouge2'],
        'rougeL': rouge_result['rougeL'],
        'bleu': bleu_result['bleu'],
        'f1': f1_result['f1']
    }
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save every 1000 steps
    learning_rate=3e-4,
    per_device_train_batch_size=2,  # Adjusted for T4 GPU
    per_device_eval_batch_size=1,   # Adjusted for T4 GPU
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    push_to_hub=False
)

# Set up trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Make predictions
def generate_predictions(trainer, test_dataset, num_samples=5):
    sample = test_dataset.select(range(num_samples))
    inputs = [prefix + q for q in sample["question"]]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=128)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return list(zip(sample["question"], predictions))

# Generate and print predictions
predictions = generate_predictions(trainer, tokenized_dataset["test"], num_samples=5)
for question, answer in predictions:
    print(f"Question: {question}")
    print(f"Predicted Answer: {answer}")
    print()
