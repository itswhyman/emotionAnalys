# WandB bypass
import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import numpy as np

# Veri yükle
dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
df = dataset['train'].to_pandas()
df = df.dropna()

# Etiket map'i
label_to_id = {'Negative': 0, 'Notr': 1, 'Positive': 2}
id_to_label = {v: k for k, v in label_to_id.items()}

df['label_int'] = df['label'].map(label_to_id)

# KÜÇÜK SUBSAMPLE (2k toplam, dengeli ~667 her sınıf – hızlı test)
df_pos = df[df['label'] == 'Positive'].sample(n=667, random_state=42)
df_notr = df[df['label'] == 'Notr'].sample(n=667, random_state=42)
df_neg = df[df['label'] == 'Negative'].sample(n=666, random_state=42)
df_sample = pd.concat([df_pos, df_notr, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Subsample shape:", df_sample.shape)

# String etiketler
df_sample['label_str'] = df_sample['label']

# Train-test split (test %20)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sample['text'].tolist(), df_sample['label_str'].tolist(), 
    test_size=0.2, random_state=42, stratify=df_sample['label']
)

print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

# Dataset'ler
train_enc = Dataset.from_dict({'text': train_texts, 'label': train_labels})
test_enc = Dataset.from_dict({'text': test_texts, 'label': test_labels})

train_enc = train_enc.map(tokenize_function, batched=True)
test_enc = test_enc.map(tokenize_function, batched=True)

# Label mapping
train_enc = train_enc.map(
    lambda examples: {'labels': [label_to_id[label] for label in examples['label']]}, 
    batched=True
)
test_enc = test_enc.map(
    lambda examples: {'labels': [label_to_id[label] for label in examples['label']]}, 
    batched=True
)

# Gereksiz sütunları sil
train_enc = train_enc.remove_columns(['text', 'label'])
test_enc = test_enc.remove_columns(['text', 'label'])

# Torch format
train_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Hazır!")

# Model
model = BertForSequenceClassification.from_pretrained(
    'dbmdz/bert-base-turkish-cased', 
    num_labels=3, 
    id2label=id_to_label, 
    label2id=label_to_id,
    ignore_mismatched_sizes=True
)

# Eğitim ayarları (hızlı: 1 epoch, batch 16)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Kısa test
    per_device_train_batch_size=16,  # Daha hızlı
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='no',  # Epoch'ta eval yapma, hız için
    save_strategy='no',
    report_to="none",
    disable_tqdm=False,
    logging_steps=20
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=test_enc,
    tokenizer=tokenizer
)

# Eğit (1-2 dk)
print("Eğitim başlıyor...")
trainer.train()

# Değerlendirme
print("Test hesaplanıyor...")
predictions = trainer.predict(test_enc)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

print("\n=== SONUÇLAR ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nDetaylı Rapor:\n", classification_report(labels, preds, target_names=['Negative', 'Notr', 'Positive']))

# Örnek
sample_text = "Bu film harikaydı, bayıldım!"
inputs = tokenizer(sample_text, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
pred_id = torch.argmax(outputs.logits).item()
print(f"\nÖrnek: '{sample_text}' → Tahmin: {id_to_label[pred_id]}")
