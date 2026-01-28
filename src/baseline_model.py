from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

class BaselineModel:
    def __init__(self, ngram_range=(1, 3), max_features=10000):
        # operate on character n-grams
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    def train(self, X_train, y_train):
        print("Vectorizing training data.")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Training Logistic Regression on {X_train_vec.shape[0]}")
        self.classifier.fit(X_train_vec, y_train)
        print("Training complete.")
        print("Vectorizing training data.")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Training Logistic Regression on {X_train_vec.shape[0]}")
        self.classifier.fit(X_train_vec, y_train)
        print("Training complete.")
        
    def evaluate(self, X_test, y_test):
        print("Vectorizing test data.")
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Predicting.")
        y_pred = self.classifier.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n=== Baseline Model Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm
        }
