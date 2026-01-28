import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_filter_data(data_dir):
    filepath = os.path.join(data_dir, 'japanese_reviews_500_preprocessed.csv')
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None, None, None
        
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records.")
        
        if 'sentiment' in df.columns:
            sentiment_map = {'positive': 1, 'negative': 0}
            df['label'] = df['sentiment'].map(sentiment_map)
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
        else:
            print("Warning: 'sentiment' column not found.")

        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        
        print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None, None
