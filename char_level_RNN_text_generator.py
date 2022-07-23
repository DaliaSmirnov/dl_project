"""
Based on the article: https://www.tensorflow.org/text/tutorials/text_generation
"""
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import random
from csv import writer
from typing import List, Union


class GRUModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


class RNNCharLevelGenerator:
    def __init__(self, data_set):
        """
        :param data_set: A Pandas series with text from a specific character/person
        """
        self.data_set = data_set
        if not isinstance(data_set, pd.Series):
            raise TypeError('Input data is not supported. It has to be a Pandas Series type')

        # preprocess raw data
        self.corpus = self.data_set.str.cat(sep='\n')
        self.corpus = self.corpus.replace('\n', ' ')
        self.corpus = self.corpus.replace('\t', ' ')
        for punctuation in ['.', '-', ',', '!', '?', '(', '—', ')']:
            self.corpus = self.corpus.replace(punctuation, f'{punctuation} ')

        self.corpus_words = self.corpus.split(' ')
        self.corpus_words = [word for word in self.corpus_words if word != '']
        self.corpus = ' '.join(self.corpus_words)
        self.vocab = sorted(set(self.corpus))
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(self.vocab), mask_token=None)
        self.chars_from_ids = tf.keras.layers.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(),
                                                           invert=True, mask_token=None)

        self.all_ids = self.ids_from_chars(tf.strings.unicode_split(self.corpus, 'UTF-8'))
        self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.all_ids)
        self.seq_length = 100
        self.examples_per_epoch = len(self.corpus)//(self.seq_length+1)
        self.sequences = self.ids_dataset.batch(self.seq_length+1, drop_remainder=True)
        self.buffer_size = 1000

    @staticmethod
    def _split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def _text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    @staticmethod
    def _create_checkpoint(character_name):
        # Directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{character_name}_gru_model")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        return checkpoint_callback

    def train(self, embedding_dim, rnn_units, batch_size, save_best_model, character_name, epochs=50):
        dataset = self.sequences.map(self._split_input_target)
        dataset = (
            dataset
            .shuffle(buffer_size=self.buffer_size)
            .batch(batch_size=batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        self.gru_model = GRUModel(vocab_size=len(self.ids_from_chars.get_vocabulary()),
                                  embedding_dim=embedding_dim, rnn_units=rnn_units)
        self.gru_model.compile(optimizer=Adam(learning_rate=1e-3),
                               loss=SparseCategoricalCrossentropy(from_logits=True))

        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='min')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='min',
                                                         min_lr=0.0001, verbose=1)
        if save_best_model:
            mc = self._create_checkpoint(character_name)
            history = self.gru_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                                         callbacks=[es, reduce_lr, mc])

        history = self.gru_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                                     callbacks=[es, reduce_lr])
        # print(f"\n\nMODEL PERPLEXITY: {tf.exp(history.history['val_loss'][-1])}")
        return history

    @staticmethod
    def _append_list_as_row(file_name: str, list_of_elem: List[Union[str, int, float]]):
        try:
            with open(file_name, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(list_of_elem)
        except:
            pass

    def hyperparameter_tune(self, save_results_locally, epochs=50, amount_iters=12):
        """
        :param save_results_locally: saves each run's best epoch as a text file locally to view results in an easy
        to read format.

        This method chooses hyperparameters randomly, trains the model and saves results if specified.
        """

        batch_sizes = [8, 16, 32, 64]
        dims = [64, 128, 256, 512, 1024]

        for i in range(1, amount_iters+1):
            batch_size = random.choice(batch_sizes)
            embedding_dim = random.choice(dims)
            rnn_units = random.choice(dims[dims.index(embedding_dim):])  # make sure rnn_units >= embedding_dim

            print(f'Current model: batch_size-{batch_size}, emb_dim-{embedding_dim}, rnn_units-{rnn_units}\n')
            history = self.train(embedding_dim=embedding_dim, rnn_units=rnn_units,
                                 batch_size=batch_size, epochs=epochs, save_best_model=False, character_name='Barney')
            best_epoch_val_loss = min(history.history["val_loss"])
            print(f'BEST EPOCH RESULT: {best_epoch_val_loss}')

            if save_results_locally:
                out_data = [f'fold no.{i}: batch_size - {batch_size}, emb_dim - {embedding_dim}, '
                            f'rnn_units - {rnn_units}, BEST_RESULT - {best_epoch_val_loss}\n'.strip('"')]
                self._append_list_as_row('hyperparameter_tracker_char_level_rnn.txt', out_data)

    def generate_text(self, seed):
        one_step_model = OneStep(self.gru_model, self.chars_from_ids, self.ids_from_chars)
        states = None
        next_char = tf.constant([seed])
        result = [next_char]

        for n in range(200):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)



"""
THE FOLLOWING CODE WAS USED TO TUNE HYPERPARAMETERS FOR THE CHARACTER BARNEY:

from project_tools import ProjectTools
tools = ProjectTools()
df = tools.clean_data('HIMYM_data.csv')
barney_series = tools.get_data_of_character(df, 'Barney')

gru = RNNCharLevelGenerator(barney_series)
gru.hyperparameter_tune(save_results_locally=True)
"""

from project_tools import ProjectTools
tools = ProjectTools()
df = tools.clean_data('HIMYM_data.csv')
barney_series = tools.get_data_of_characters(df, ['Barney'])
gru = RNNCharLevelGenerator(barney_series)
gru.train(embedding_dim=512, rnn_units=1024, batch_size=32, save_best_model=True, character_name='Barney')




