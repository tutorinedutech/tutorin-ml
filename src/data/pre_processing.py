from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from gensim.models import Word2Vec

import numpy as np
import gc

"""def train_word2vec(documents, embedding_dim):
    model = Word2Vec(documents, min_count=1, vector_size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors

def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def word_embed_meta_data(documents, embedding_dim):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(' '.join(documents))
    word_vectors = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vectors, embedding_dim)
    del word_vectors
    gc.collect()
    return embedding_matrix"""

def create_train_dev(df, max_sequence_length, validation_split_ratio):
    prompt = df['Prompt'].astype(str)
    kriteria = df['combined'].astype(str)
    labels = df['Skor Label']

    prompt = df['Prompt'].tolist()
    kriteria = df['combined'].tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(prompt + kriteria)
    sentence_pair = [(x1, x2) for x1, x2 in zip(prompt, kriteria)]

    prompt = [x[0] for x in sentence_pair]
    kriteria = [x[1] for x in sentence_pair]

    prompt = tokenizer.texts_to_sequences(prompt)
    kriteria = tokenizer.texts_to_sequences(kriteria)
    '''leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(prompt, kriteria)]'''
    
    padded_data_1 = pad_sequences(prompt, maxlen=max_sequence_length)
    padded_data_2 = pad_sequences(kriteria, maxlen=max_sequence_length)
    labels = (np.array(labels)/2.5) - 1.0

    #leaks = np.array(leaks)
    
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    train_data_1_shuffled = padded_data_1[shuffle_indices]
    train_data_2_shuffled = padded_data_2[shuffle_indices]
    train_labels_shuffled = labels[shuffle_indices]
    #leaks_shuffled = leaks[shuffle_indices]
    
    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))
    
    del padded_data_1
    del padded_data_2
    gc.collect()
    
    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    #leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    
    return train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val

def create_test_data(df, max_sequence_length):
    prompt = df['Prompt'].astype(str)
    kriteria = df['combined'].astype(str)

    prompt = df['Prompt'].tolist()
    kriteria = df['combined'].tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(' '.join(prompt+kriteria))
    sentence_pair = [(x1, x2) for x1, x2 in zip(prompt, kriteria)]

    prompt = [x[0] for x in sentence_pair]
    kriteria = [x[1] for x in sentence_pair]

    seq_prompt = tokenizer.texts_to_sequences(prompt)
    seq_kriteria = tokenizer.texts_to_sequences(kriteria)

    padded_prompt = pad_sequences(seq_prompt, maxlen=max_sequence_length)
    padded_kriteria = pad_sequences(seq_kriteria, maxlen=max_sequence_length)

    return padded_prompt, padded_kriteria