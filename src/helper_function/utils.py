import pandas as pd
import nltk

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def combine_data(df):
    df['combined'] = 