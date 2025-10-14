import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

print("="*60)
print("          DARK PATTERN MODEL EVALUATION SCRIPT")
print("="*60)

# --- This setup must match your training script ---

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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# --- Main Evaluation Logic ---

try:
    # 1. Load the tokenizer and the final trained model
    print("\n[INFO] Loading final trained model and tokenizer...")
    model_path = "./final_dark_pattern_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    print("[SUCCESS] Model loaded successfully.")

    # 2. Load the full dataset to get the validation set
    print("\n[INFO] Loading the full dataset to create the test set...")
    df = pd.read_csv('combined_dark_patterns_FULL.csv')
    df.dropna(subset=['text', 'category'], inplace=True)

    categories = sorted(df['category'].unique().tolist())
    label2id = {cat: i for i, cat in enumerate(categories)}
    id2label = {i: cat for i, cat in enumerate(categories)}

    df['label_id'] = df['category'].map(label2id)
    df['processed_text'] = df['text'].apply(clean_text)

    # 3. Create the exact same train/test split to get the validation data
    # Using the same random_state ensures we get the same split as during training
    _, val_texts, _, val_labels = train_test_split(
        df['processed_text'].tolist(),
        df['label_id'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label_id']
    )
    print(f"[SUCCESS] Test set created with {len(val_texts)} samples.")

    # 4. Tokenize the validation data
    print("\n[INFO] Tokenizing the test set...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    val_dataset = DarkPatternDataset(val_encodings, val_labels)
    print("[SUCCESS] Tokenization complete.")


    # 5. Use the Trainer to make predictions on the validation set
    print("\n[INFO] Running model predictions on the test set...")
    # We need a dummy TrainingArguments object for the Trainer
    dummy_args = TrainingArguments(output_dir="./temp_eval", report_to="none")
    trainer = Trainer(model=model, args=dummy_args)

    predictions_output = trainer.predict(val_dataset)
    y_preds = predictions_output.predictions.argmax(axis=1)
    y_true = predictions_output.label_ids
    print("[SUCCESS] Prediction complete.")


    # 6. Generate and print the performance report
    print("\n" + "="*60)
    print("                OFFICIAL MODEL PERFORMANCE REPORT")
    print("="*60)

    # Overall Accuracy
    accuracy = accuracy_score(y_true, y_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    print("-"*60)

    # Detailed Classification Report (Precision, Recall, F1-Score per category)
    print("\nDetailed Classification Report:\n")
    # Use the category names (labels) in the report
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(y_true, y_preds, target_names=target_names)
    print(report)
    print("="*60)
    print("    COPY THE ENTIRE OUTPUT ABOVE THIS LINE AND PASTE IT")
    print("="*60)


except FileNotFoundError:
    print("\n[ERROR] The 'final_dark_pattern_model' folder or 'combined_dark_patterns_FULL.csv' was not found.")
    print("Please make sure you have successfully run the `create_full_dataset.py` and `dark_pattern_detector.py` scripts first.")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")

