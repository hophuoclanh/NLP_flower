import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load the News Category dataset from a JSON file."""
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Combine headline and short description
    df['text'] = df['headline'] + " " + df['short_description']
    df = df.dropna(subset=['text', 'category'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    
    return df, label_encoder

def split_data(df):
    """Split the dataset into train, validation, and test sets."""
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    return train, val, test
