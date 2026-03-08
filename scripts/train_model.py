import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from preprocess import clean_text
import os

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AnxietyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'dataset', 'anxiety_dataset.csv')
    df = pd.read_csv(dataset_path)
    
    # Preprocess text
    df['text'] = df['text'].apply(clean_text)
    
    # Map labels to integers
    label_map = {
        'Low Anxiety': 0,
        'Moderate Anxiety': 1,
        'High Anxiety': 2
    }
    df['label'] = df['label'].map(label_map)
    
    # Parameters
    MAX_LEN = 64
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Dataloaders
    train_dataset = AnxietyDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model = model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save model
    model_dir = os.path.join(base_dir, '..', 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_save_path = os.path.join(model_dir, 'anxiety_model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
