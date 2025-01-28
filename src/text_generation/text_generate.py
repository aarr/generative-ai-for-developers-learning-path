import os
import time
import warnings
import numpy as np
import tensorflow as tf
from typing import Callable

import MyModel as mdl
import OneStep as osmdl
import CustomTraining as csmdl

# 文字列からbyteコードへ変換する関数を生成
def generate_ids_from_chars(vocab:str) -> Callable[[str], object]:
    return tf.keras.layers.StringLookup(
        vocabulary=list(vocab),
        mask_token=None
        )

# byteコードから文字列へ変換する関数を生成
def generate_chars_from_ids(ids_from_chars: Callable[[str], object]) -> Callable[[str], object]:
    return tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )

# byteコードからテキストへ変換する関数を生成
def generate_text_from_ids(ids_from_chars: Callable[[str], object]) -> Callable[[str], object]:
    return lambda ids : tf.strings.reduce_join(generate_chars_from_ids(ids_from_chars)(ids) ,axis=-1)

#
# メイン
#
def main():
    # GPUアクセスチェック
    # Check for TensorFlow GPU access
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    # See TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")


    # ロジック
    # シェイクスピアの作品をロードし、使用している文字をカウント
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )

    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    print(f"Length of file:{len(text)} characters")
    print(text[:250])

    vocab = sorted(set(text))
    print(f"{len(vocab)} unique characters")


    # 文字列を文字ごとに分割
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    print(f"char:{chars}")

    ids_from_chars = generate_ids_from_chars(vocab)
    # 文字をバイトコードに変換
    ids = ids_from_chars(chars)
    print(f"id type:{type(ids)}")
    print(f"char->id:{ids}")

    # バイトコードを文字に変換
    chars_from_ids = generate_chars_from_ids(ids_from_chars)
    chars = chars_from_ids(ids)
    print(f"char type:{type(chars)}")
    print(f"id->char:{chars}")

    # バイドコードからテキストに変換
    text_from_ids = generate_text_from_ids(ids_from_chars)
    text_ex = text_from_ids(ids)
    print(f"text_ex type:{type(text_ex)}")
    print(f"id->text:{text_ex}")


    all_ids = ids_from_chars(tf.strings.unicode_split(text, input_encoding='UTF-8'))
    print(f"all_ids:{all_ids}")

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    print(f"ids_dataset:{ids_dataset}")

    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().decode('utf-8'))
    
    seq_length = 100
    example_per_epoch = len(text) // (seq_length + 1)
    print(f"text length:{len(text)}")
    print(f"example_per_epoch:{example_per_epoch}")

    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
    for seq in sequences.take(5):
        print(f"seq : {chars_from_ids(seq)}")
    
    def split_input_target(sequences):
        input_text = sequences[:-1]
        target_text = sequences[1:]
        return input_text, target_text

    split_input_target(list('Tensorflow'))
    dataset = sequences.map(split_input_target)
    for input_example, target_example in dataset.take(1):
        print('Input  : ', text_from_ids(input_example).numpy())
        print('Target : ', text_from_ids(target_example).numpy())
    

    # Build The Model
    print("\n", "-" * 10, "Build Model", "-" * 10)
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    # dataset再定義
    dataset = (
        dataset.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    print(f"dataset : {dataset}")

    vocab_size = len(vocab)     # Length of the vocabulary in chars
    embedding_dim = 256         # The embedding dimension
    rnn_unit = 1024             # Number of RNN units

    # Model生成
    model = mdl.MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_unit=rnn_unit,
    )
    # Modelトレーニング
    for input_example_batch, target_examle_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(f"input_example_batch : {input_example_batch}")
        print(f"target_example_batch : {target_examle_batch}")
        print(
            "MODEL :",
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)"
        )
    
    sample_indices = tf.random.categorical(
        example_batch_predictions[0], num_samples=1
    )
    sample_indices = tf.squeeze(sample_indices, axis=1).numpy()
    print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
    print("Next Char Predictions:\n", text_from_ids(sample_indices).numpy())

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_examle_batch, example_batch_predictions)
    print(
        "Prediction Shape: ",
        example_batch_predictions.shape,
        "# (batch_size, sequence_length, vocab_size)"
    )
    print("Mean Loss:      ", example_batch_mean_loss)
    print(f"Exp : {tf.exp(example_batch_mean_loss).numpy()}")

    # Modelのトレーニング
    model.compile(optimizer="adam", loss=loss)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )
    EPOCH = 10
    history = model.fit(dataset, epochs=EPOCH, callbacks=[checkpoint_callback])
    print(f"history : {history}")

    one_step_model = osmdl.OneStep(model, chars_from_ids, ids_from_chars)
    start = time.time()
    states = None
    next_char = tf.constant(["ROMEO:"])
    # バッチ化
    #next_char = tf.constant(["ROMEO:", "ROMEO:", "ROMEO:", "ROMEO:", "ROMEO:"])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(
            next_char,
            states
        )
        result.append(next_char)
    
    result = tf.strings.join(result)
    end = time.time()

    print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
    # バッチ化
    # for s in result:
    #     print(s.numpy().decode("utf-8"))
    print("\nRun time:", end - start)

    print("\n", "-" * 10, "Model Saved & Load", "-" * 10)
    # Model保存＆ロード
    tf.saved_model.save(one_step_model, "one_step")
    one_step_reloaded = tf.saved_model.load("one_step")

    states = None
    next_char = ["ROMEO:"]
    result = [next_char]

    for n in range(100):
        next_char, states = one_step_reloaded.generate_one_step(
            next_char,
            states
        )
        result.append(next_char)

    print(tf.strings.join(result)[0].numpy().decode("utf-8"))

    # Customized Traingin
    print("\n", "-" * 10, "Model Customized Training", "-" * 10)
    model = csmdl.CustomTraining(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_unit=rnn_unit
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    model.fit(dataset, epochs=1)

    mean = tf.metrics.Mean()
    for epoch in range(EPOCH):
        start = time.time()

        mean.reset_state()
        for batch_n, (inp, target) in enumerate(dataset):
            logs = model.train_step([inp, target])
            loss = logs["loss"]
            mean.update_state(loss)

            if batch_n % 50 == 0:
                template = (
                    f"Epoch {epoch + 1} Batch {batch_n} Loss {loss:.4f}"
                )
                print(template)

        if (epoch + 1) % 5 == 0:
           model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print()
        print(f"Epoch {epoch + 1} Loss {mean.result().numpy():.4f}")
        print(f"Time taken for 1 epoch {time.time() - start:.2f} sec")
        print("-" * 80)

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

# debugする為には実行が必要
# main()
