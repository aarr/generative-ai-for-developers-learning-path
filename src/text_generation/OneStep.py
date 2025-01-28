import tensorflow as tf

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__(self)
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        
        # UNKを性せされないようにする
        skip_ids = self.ids_from_chars(['UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # 文字列からTokenIDへ変換
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Model実行
        # 確率値算出
        predicted_logits, states = self.model(
            inputs=input_ids,
            states=states,
            return_state=True
        )
        # 最後の予測のみ利用
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask
        
        #  確率値からtokenIDへ変換
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=1)

        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states
