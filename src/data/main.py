from src.data.pre_processing import train_word2vec, create_embedding_matrix, create_train_dev
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('data_source/processed/clean_data.csv')

documents = df['Prompt'].tolist() + df['combined'].tolist()
prompt = df['Prompt'].tolist()
kriteria = df['combined'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(" ".join(documents))

sentence_pair = [(x1, x2) for x1, x2 in zip(prompt, kriteria)]
leaks = create_train_dev(tokenizer, sentence_pair)
print(leaks)