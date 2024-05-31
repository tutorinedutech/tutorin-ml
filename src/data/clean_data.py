import os
import sys
from src.utils.helper_function import load_data, combine_data, clean_data, stopword_removal

data_path = 'https://raw.githubusercontent.com/tutorinedutech/tutorin-ml/main/data_source/raw/Data.csv'
stop_words_path = 'https://raw.githubusercontent.com/tutorinedutech/tutorin-ml/main/data_source/raw/stopwordbahasa.csv'
df = load_data(data_path)

df = combine_data(df)

df['combined'] = df['combined'].apply(lambda x: clean_data(x))
df['Prompt'] = df['Prompt'].apply(lambda x: clean_data(x))

df['combined'] = df['combined'].apply(str).apply(lambda x: stopword_removal(x))
df['Prompt'] = df['Prompt'].apply(str).apply(lambda x: stopword_removal(x))

clean_df = df[['Prompt', 'combined', 'Skor Label']]

clean_df.to_csv('clean_data.csv')







