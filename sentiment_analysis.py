import os
import csv
import re
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch

# Function to get train and test lists
def get_train_test_list(train, test):
    try:
        X = pd.read_csv(train)
        x = pd.read_csv(test)
    except pd.errors.ParserError as e:
        print(f"Error: {e}")
        return [], []

    X_sampled = X.sample(n=4000, random_state=1)
    x_sampled = x.sample(n=2000, random_state=1)

    X_train = X_sampled['Text']
    y_train = X_sampled['Sentiment']

    X_test = x_sampled['Text']
    y_test = x_sampled['Sentiment']

    train_list = []
    test_list = []

    def clean_text(text):
        return re.sub(r'[^A-Za-z0-9\s]', '', text)

    for text_train, text_test in zip(X_train, X_test):
        cleaned_text_train = clean_text(text_train)
        cleaned_text_test = clean_text(text_test)
        train_list.append(cleaned_text_train)
        test_list.append(cleaned_text_test)
    
    return y_train, y_test, train_list, test_list

# Function to encode texts using tokenizer
def tokenizer_encoding_labels(train_list, test_list, y_train, y_test):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_list, truncation=True, padding=True)
    test_encodings = tokenizer(test_list, truncation=True, padding=True)
    train_labels = [i for i in y_train]
    test_labels = [i for i in y_test]
    return train_encodings, test_encodings, train_labels, test_labels

# Dataset class for sentiment analysis
class SentimentAnalysis(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to make predictions
def prediction(test_dataset, device, model):
    all_predictions = []
    all_labels = []
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)

            predicted_labels = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    return all_predictions, all_labels

# Function to calculate metrics
def metrics(all_labels, all_predictions):
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

# Main function
def main():

    # Set CUDA_LAUNCH_BLOCKING to 1 for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set up DagsHub as the remote tracking server
    MLFLOW_TRACKING_URI = "https://dagshub.com/karmakaragradwip02/sentiment_analysis_BERTMODEL.mlflow"
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'karmakaragradwip02'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '9ccb0f28354fcca6469017b32544fa0704b9c343'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    y_train, y_test, train_list, test_list = get_train_test_list('train.csv', 'test.csv')
    train_encodings, test_encodings, train_labels, test_labels = tokenizer_encoding_labels(train_list, test_list, y_train, y_test)
    train_dataset = SentimentAnalysis(train_encodings, train_labels)
    test_dataset = SentimentAnalysis(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-6, eps=2e-7, weight_decay=0.001)

    with mlflow.start_run() as run:
        try:
            # Log hyperparameters
            mlflow.log_param("learning_rate", 5e-6)
            mlflow.log_param("epsilon", 2e-7)
            mlflow.log_param("weight_decay", 0.01)
            mlflow.log_param("batch_size", 4)

            # Training loop
            for epoch in range(3):
                print("Epoch:", epoch)
                total_loss = 0.0
                for batch in train_loader:
                    optim.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Average Loss after epoch {epoch}: {avg_loss}")
                mlflow.log_metrics({f"Loss_epoch_{epoch}": avg_loss})
            mlflow.pytorch.log_model(model, "model")
            model.eval()
            all_predictions, all_labels = prediction(test_dataset, device, model)
            accuracy, precision, recall, f1 = metrics(all_labels, all_predictions)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1:", f1)

            mlflow.log_metric("Accuracy:", accuracy)
            mlflow.log_metric("Precision:", precision)
            mlflow.log_metric("Recall:", recall)
            mlflow.log_metric("F1:", f1)

        except Exception as e:
            print(f"Exception during training: {e}")
        finally:
            mlflow.end_run()

if __name__=="__main__": 
    main()