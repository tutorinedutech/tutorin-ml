import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def combine_data(df):
    df['combined'] = df.apply(lambda row:f"Jenis Kelamin: {row['Jenis Kelamin']}" + 
                              f"Provinsi: {row['Provinsi']} Kabupaten/Kota: {row['Kabupaten/Kota']}" + 
                              f"Interest: {row['Interest']} Bahasa: {row['Bahasa']} Kriteria: {row['Kriteria']}")
    return df

def clean_data(text):
    text = text.lower() #change text to lower
    text = re.sub(r'\s+', ' ', text).strip() #remove extra whitespace
    return text

def stopword_removal(text):
    stopword_path = 'https://raw.githubusercontent.com/tutorinedutech/tutorin-ml/main/data_source/raw/stopwordbahasa.csv'
    stopword_list = pd.read_csv(stopword_path).values.tolist()
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopword_list]
    combined_text = " ".join(tokens_without_sw)
    return combined_text

def split_data(df, random_state):
    df_train , df_tmp = train_test_split(df, test_size=0.3, shuffle=True, random_state=random_state)
    df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=random_state)
    return df_train, df_val, df_test 
