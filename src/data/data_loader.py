import pandas as pd
import tensorflow as tf

class DataLoader:
    def __init__(self, df,  train_split, buffer_size):
        self.df = df
        self.train_split = train_split
        self.buffer_size = buffer_size

    def get_tensor_data(self):
        X_df = self.df.drop('Skor Label', axis=1)
        y_df = self.df['Skor Label']
        ds = tf.data.Dataset.from_tensor_slices((X_df['Prompt'], X_df['combined'], y_df.values))
        return ds
    
    def get_prefetch_data(self, ds):
        ds = ds.map(lambda prompt, kriteria, y: {'Prompt' : prompt, 'Kriteria' : kriteria, 'label':y})
        return ds.prefetch(1)    
   
    def split_data(self, ds):
        train_ds_size = int(self.train_split*self.df.shape[1])
        remain_ds_size = int((1-self.train_split)*self.df.shape[1])
        ds = ds.shuffle(self.buffer_size)
        train_ds = ds.take(train_ds_size)
        remain_ds = ds.skip(train_ds_size)
        val_ds = remain_ds.take(remain_ds_size)
        test_ds = remain_ds.skip(remain_ds_size)
        return train_ds, val_ds, test_ds
    
class Preprocess:
     def __init__(self, batch_size, num_batches, AUTOTUNE):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.AUTOTUNE = AUTOTUNE

     def change_range(self, x):
        x = (x/2.5)-2
        return x
    
    def preprocessing_data(self, ds):
        ds = ds.map(
            lambda z: (
                [z['Prompt'], z['Kriteria']],
                [tf.cast(self.change_range(z['label']), tf.float32)]
            ),
            num_parallel_calls = self.AUTOTUNE
        )
        ds = ds.batch(self.batch_size)
        ds = ds.take(self.num_batches)
        ds = ds.prefetch(self.AUTOTUNE)
        return ds
