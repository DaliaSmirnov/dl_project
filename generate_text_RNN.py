from project_tools import ProjectTools
import tensorflow as tf
from keras.models import load_model

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



tools = ProjectTools()
df = tools.clean_data('HIMYM_data.csv')
data_set = tools.get_data_of_characters(df, ['Barney'])
WEIGHTS_PATH = './training_checkpoints/ckpt_Barney_gru_model'


corpus = data_set.str.cat(sep='\n')
corpus = corpus.replace('\n', ' ')
corpus = corpus.replace('\t', ' ')
for punctuation in ['.', '-', ',', '!', '?', '(', '—', ')']:
    corpus = corpus.replace(punctuation, f'{punctuation} ')

corpus_words = corpus.split(' ')
corpus_words = [word for word in corpus_words if word != '']
corpus = ' '.join(corpus_words)
vocab = sorted(set(corpus))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=512, rnn_units=1024)

model.load_weights(WEIGHTS_PATH)


def generate_text(seed):
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = None
    next_char = tf.constant([seed])
    result = [next_char]

    for n in range(200):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)


generate_text('Have you met')