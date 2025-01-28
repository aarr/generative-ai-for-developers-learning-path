import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_unit):
        super().__init__(self)

        # TODO create embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # TODO create GRU layer
        self.gru = tf.keras.layers.GRU(
            rnn_unit,
            return_sequences=True,
            return_state=True
        )
        # TODO Finally connect it with a dense layer
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)

        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
