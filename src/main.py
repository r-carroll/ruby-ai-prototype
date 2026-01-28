from src.data_loader import load_and_filter_data
from src.baseline_model import BaselineModel
from src.bert_model import BERTClassifier

def main():
    print("Starting NLP pipeline")
    
    data_dir = "data"
    train_df, val_df, test_df = load_and_filter_data(data_dir)
    
    baseline = BaselineModel()
    
    X_train = train_df['clean_text'].astype(str)
    y_train = train_df['label']
    
    X_test = test_df['clean_text'].astype(str)
    y_test = test_df['label']
    
    baseline.train(X_train, y_train)
    baseline.evaluate(X_test, y_test)
    
    print("\n--- BERT Model ---")
    bert = BERTClassifier()
    
    train_loader = bert.create_data_loader(train_df, batch_size=16)
    val_loader = bert.create_data_loader(val_df, batch_size=16)
    test_loader = bert.create_data_loader(test_df, batch_size=16)
    bert.train(train_loader, val_loader, epochs=3)
    
    print("Evaluating")
    acc, metrics = bert.evaluate(test_loader)
    print(f"Accuracy: {acc}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    bert.save_model("models/bert_model")
    bert.export_onnx("models/model.onnx")
    
    print("Pipeline complete")

if __name__ == "__main__":
    main()
