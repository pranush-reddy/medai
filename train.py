import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Verify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Create Dataset with 4 Disease Classes
    def create_dataset():
        disease_categories = {
            0: ("Diabetes", [
                "Type 2 diabetes with HbA1c 8.5%",
                "Patient presents with polyuria and polydipsia",
                "Diabetic neuropathy in lower extremities",
                "Elevated glucose levels (220 mg/dL)",
                "Insulin resistance with weight gain",
                "Diabetic ketoacidosis requiring hospitalization"
            ]),
            1: ("Heart Disease", [
                "Chest pain with elevated troponin levels",
                "History of myocardial infarction",
                "ECG shows ST segment elevation",
                "Diagnosed with coronary artery disease",
                "Patient with congestive heart failure",
                "Echocardiogram shows reduced ejection fraction",
                "Atrial fibrillation with rapid ventricular response"
            ]),
            2: ("Asthma", [
                "Wheezing and shortness of breath",
                "Asthma exacerbation requiring albuterol",
                "Chronic bronchospasm",
                "Peak flow variability >20%",
                "Severe persistent asthma with nighttime symptoms",
                "Patient requires corticosteroid inhaler"
            ]),
            3: ("Kidney Disease", [
                "Chronic kidney disease stage 3",
                "Elevated creatinine (2.4 mg/dL)",
                "Proteinuria on urinalysis",
                "Patient requires hemodialysis",
                "End-stage renal disease with electrolyte imbalances",
                "Hypertensive nephropathy causing kidney dysfunction"
            ])
        }

        texts = []
        labels = []
        for label, (name, examples) in disease_categories.items():
            texts.extend(examples)
            labels.extend([label] * len(examples))
        
        return pd.DataFrame({'text': texts, 'label': labels})

    df = create_dataset()
    logger.info(f"Dataset created with {len(df)} examples")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")

    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    class MedicalDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # 5. Initialize tokenizer and data loaders
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

    def create_data_loader(df, tokenizer, batch_size=8, shuffle=True):
        return DataLoader(
            MedicalDataset(
                df.text.to_numpy(),
                df.label.to_numpy(),
                tokenizer
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    BATCH_SIZE = 8
    train_data_loader = create_data_loader(train_df, tokenizer, BATCH_SIZE)
    val_data_loader = create_data_loader(val_df, tokenizer, BATCH_SIZE, shuffle=False)

    # 6. Model Setup
    model = BertForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-base-cased-v1.1',
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    logger.info("Model loaded successfully")

    # 7. Training Configuration
    EPOCHS = 10  # Increased for better learning
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    # Class weights (adjusted for balance)
    class_weights = torch.tensor([1.0, 1.2, 1.1, 1.3], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 8. Training Loop
    best_accuracy = 0
    best_model_path = 'best_model_state.bin'

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch in train_data_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_acc = correct_predictions.double() / len(train_data_loader.dataset)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {train_acc:.4f}")

        # Save best model
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with accuracy: {best_accuracy:.4f}")

    # 9. Load and evaluate best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification report
    class_names = ["Diabetes", "Heart Disease", "Asthma", "Kidney Disease"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

if __name__ == '__main__':
    main()
