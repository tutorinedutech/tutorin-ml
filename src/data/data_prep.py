import tensorflow as tf
import pandas as pd

def load_data(data_path):
    '''
    Defines for load dataset

    Args:
        data_path: A data path where the dataset is located.
    
    returns:
        similar_df: The pandas dataframe for user which similarity score with mentor.
        mentor_df: The pandas dataframe for all mentor corpus dataset.
    '''
    similar_df = pd.read_excel(data_path, sheet_name='similar')
    mentor_df = pd.read_excel(data_path, sheet_name='mentor')
    return similar_df, mentor_df

def get_tensor_data(similar_df, mentor_df):
    '''
    the function used to obtain the dataset tensor.

    Args:
        similar_df: The pandas dataframe for user which similarity score with mentor.
        mentor_df: The pandas dataframe for all mentor corpus dataset.

    returns:
        similar_ds: The tensor dataset with just user_id, mentor_id, and kriteria_mentor_user
        mentor_ds: The tensor dataset with just mentor_id
    '''
    similar_ds = tf.data.Dataset.from_tensor_slices(dict(similar_df[['user_id', 'mentor_id', 'kriteria_mentor_user', 'kriteria_mentor','similarity']]))
    mentor_ds = tf.data.Dataset.from_tensor_slices(dict(mentor_df[['mentor_id']]))
    similar_ds = similar_ds.map(lambda x: {'user_id': tf.as_string(x['user_id']),
                                           'mentor_id': tf.as_string(x['mentor_id']),
                                           'kriteria_mentor_user': x['kriteria_mentor_user'],
                                           'kriteria_mentor': x['kriteria_mentor'],
                                           'similarity': float(x['similarity'])})
    mentor_ds = mentor_ds.map(lambda x: tf.as_string(x['mentor_id']))
    return similar_ds, mentor_ds

def split_data(similar_ds, train_split, random_seed):
    '''
    This function for separate train and test data.

    Args:
        similar_ds: Tensor dataset with similarity score
        train_plit: size of split

    returns:
        cached_train: Tensor dataset for train.
        cached_test: Tensor dataset for test.
    '''

    length_data = len(list(similar_ds))
    train_size = int(length_data*train_split)
    test_size = length_data-train_size

    tf.random.set_seed(random_seed)
    shuffled = similar_ds.shuffle(100, seed=random_seed, reshuffle_each_iteration=False)

    train = shuffled.take(train_size)
    test = shuffled.skip(train_size).take(test_size)

    cached_train = train.batch(4)
    cached_test = test.batch(4)
    return cached_train, cached_test