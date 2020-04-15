import tensorflow as tf
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import numpy as np

plotly.offline.init_notebook_mode(connected=True)


def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix=None):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=False)

        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix=None):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=False)

        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss)


def evaluate(inputs, encoder, decoder):
    attention_plot = np.zeros((max_length["canto"], max_length["stdch"]))
    sentence = ''
    for i in inputs[0]:
        if i == 0:
            break
        sentence = sentence + rev_inp_token_ind[i] + ' '
    sentence = sentence[:-1]
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tar_token_ind["START_"]], 0)

    for t in range(max_length["canto"]):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += rev_tar_token_ind[predicted_id] + " "
        if rev_tar_token_ind[predicted_id] == "_END":
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence, attention_plot


def predict_val_sentence(k):
    actual_sent = ''
    inp = input_tensor_val[k]
    out = target_tensor_val[k]
    inp = np.expand_dims(inp, 0)
    result, sentence, attention_plot = evaluate(inp, encoder, decoder)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    for i in out:
        if i == 0:
            break
        actual_sent = actual_sent + rev_tar_token_ind[i] + ' '
    print("Actual translation: {}".format(actual_sent))
    attention_plot = attention_plot[:len(result.split(' ')), 1:len(sentence.split(' '))-1]
    sentence, result = sentence.split(' '), result.split(' ')
    sentence = sentence[1:-1]
    result = result[:-2]

    # use plotly to generate the heat map
    trace = go.Heatmap(z = attention_plot, x = sentence, y = result, colorscale='Reds')
    data=[trace]
    iplot(data)
