import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

class JapaneseReviewsDataset(Dataset):
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
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, model_name='tohoku-nlp/bert-base-japanese-v3', num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        self.model.to(self.device)
        
    def create_data_loader(self, df, batch_size=16, max_len=128):
        ds = JapaneseReviewsDataset(
            texts=df['clean_text'].to_numpy(),
            labels=df['label'].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=max_len
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            
            self.model.train()
            total_loss = 0
            
            for d in train_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
            avg_train_loss = total_loss / len(train_loader)
            print(f"Average train loss: {avg_train_loss}")
            
            val_acc, _ = self.evaluate(val_loader)
            print(f"Validation Accuracy: {val_acc}")
            
    def evaluate(self, data_loader):
        self.model.eval()
        
        predictions = []
        real_values = []
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                real_values.extend(targets.cpu().tolist())
                
        metrics = {
            "accuracy": accuracy_score(real_values, predictions),
            "precision": precision_score(real_values, predictions, average='binary'),
            "recall": recall_score(real_values, predictions, average='binary'),
            "confusion_matrix": confusion_matrix(real_values, predictions)
        }
        
        return metrics['accuracy'], metrics

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def export_onnx(self, path):
        print(f"Exporting model to ONNX at {path}")
        self.model.eval()
        
        dummy_input = self.tokenizer(
            "これはテストです",
            return_tensors="pt"
        )
        input_ids = dummy_input['input_ids'].to(self.device)
        attention_mask = dummy_input['attention_mask'].to(self.device)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.onnx.export(
            self.model,
            (input_ids, attention_mask),
            path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            },
            opset_version=17
        )
