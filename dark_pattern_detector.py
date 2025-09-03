# ==============================================================================
# Final Dark Pattern Detector - Training on the Full Dataset
# ==============================================================================
import os
# Force CPU usage - disable MPS and CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure CPU usage
if torch.backends.mps.is_available():
    torch.backends.mps.is_available = lambda: False

# ----------------------------
# Dataset class
# ----------------------------
class DarkPatternDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ----------------------------
# Metrics function for evaluation
# ----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ----------------------------
# Load and Prepare the REAL Dataset
# ----------------------------
print("Loading full dataset from combined_dark_patterns.csv...")
df = pd.read_csv('combined_dark_patterns_FULL.csv')
df.dropna(subset=['text', 'category'], inplace=True) # Remove any empty rows

# Create mappings from category names to numbers (IDs) and back
categories = df['category'].unique().tolist()
label2id = {cat: i for i, cat in enumerate(categories)}
id2label = {i: cat for i, cat in enumerate(categories)}

df['label_id'] = df['category'].map(label2id)

texts = df['text'].tolist()
labels = df['label_id'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print("Dataset loaded and split.")

# ----------------------------
# Tokenizer & Model
# ----------------------------
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(categories), # Use the number of actual categories
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# ----------------------------
# Prepare datasets for Trainer
# ----------------------------
print("Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = DarkPatternDataset(train_encodings, train_labels)
val_dataset = DarkPatternDataset(val_encodings, val_labels)
print("Tokenization complete.")

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results_full",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Use a slightly larger batch size
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_full',
    logging_steps=50,
    eval_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch", # Save at the end of each epoch
    load_best_model_at_end=True,
    report_to="none",
    use_cpu=True,  # Force CPU usage
    dataloader_pin_memory=False  # Disable pin memory for CPU
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ----------------------------
# Train the Model
# ----------------------------
print("🚀 Training model on the full dataset... (This will take 15-45 minutes)")
trainer.train()
print("🎉 Training complete!")

# ----------------------------
# Save the Final Model
# ----------------------------
trainer.save_model("./final_dark_pattern_model")
tokenizer.save_pretrained("./final_dark_pattern_model")
print("✅ Model saved to './final_dark_pattern_model'")

# ----------------------------
# Test Predictions
# ----------------------------
print("\n" + "="*50)
print("🧪 TESTING PREDICTIONS")
print("="*50)

# Use the trainer's predict method for a proper evaluation on the test set
predictions = trainer.predict(val_dataset)
predicted_labels = [id2label[p] for p in predictions.predictions.argmax(-1)]

# Print a few examples
for i in range(5):
    print(f"\nText:       '{val_texts[i]}'")
    print(f"Actual:     {id2label[val_labels[i]]}")
    print(f"Prediction: {predicted_labels[i]}")