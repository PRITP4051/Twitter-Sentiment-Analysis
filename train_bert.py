import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ---------------- LOAD DATA ----------------
df = pd.read_csv("labeled_data.csv")

# 🔥 Increase dataset size (balanced)
df = df.sample(6000, random_state=42)

df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)
df = df[df['clean_text'].str.strip() != ""]

# ---------------- SPLIT ----------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# ---------------- TOKENIZER ----------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(example):
    return tokenizer(
        example['text'],
        padding='max_length',
        truncation=True,
        max_length=64   # ✅ reduced for speed + efficiency
    )

# ---------------- DATASET ----------------
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ---------------- MODEL ----------------
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",   # ✅ enable evaluation
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ---------------- TRAIN ----------------
trainer.train()

# ---------------- EVALUATION ----------------

# 🔥 TEST SET
test_predictions = trainer.predict(test_dataset)
y_test_pred = np.argmax(test_predictions.predictions, axis=1)
y_test_true = test_labels.values

test_acc = accuracy_score(y_test_true, y_test_pred)
precision = precision_score(y_test_true, y_test_pred, average='weighted')
recall = recall_score(y_test_true, y_test_pred, average='weighted')
f1 = f1_score(y_test_true, y_test_pred, average='weighted')

# 🔥 TRAIN SET (NEW ADDITION)
train_predictions = trainer.predict(train_dataset)
y_train_pred = np.argmax(train_predictions.predictions, axis=1)
y_train_true = train_labels.values

train_acc = accuracy_score(y_train_true, y_train_pred)

# ---------------- PRINT ----------------
print("\n🔥 BERT Results:")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# ---------------- SAVE MODEL ----------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

# ---------------- SAVE RESULTS ----------------
bert_results = pd.DataFrame([{
    "model": "BERT",
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

bert_results.to_csv("bert_result.csv", index=False)

print("\n✅ BERT results saved!")