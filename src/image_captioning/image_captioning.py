import os
import time
from textwrap import wrap

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import Input
# 利用するレイヤーを全てimport
from tensorflow.keras.layers import (
    GRU,
    Add,
    AdditiveAttention,
    Attention,
    Concatenate,
    Dense,
    Embedding,
    LayerNormalization,
    Reshape,
    StringLookup,
    TextVectorization
)
import ssl

# ハイパーパラメータ
# 精度/速度（accuracy/speed）を変えるためのパラメータ
VOCAB_SIZE = 20000
ATTENTION_DIM = 512
WORD_EMBEDDING_DIM = 128

FEATURE_EXTRACTOR = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights="imagenet"
)

IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNEL = 3
FEATURES_SHAPE = (8, 8, 1536)


GCS_DIR = "gs://asl-public/data/tensorflow_datasets/"
BUFFER_SIZE = 1000

MAX_CAPTION_LEN = 64

def main():
    print("START")
    # 証明書無効化
    # 無効化しないとエラーになるので暫定対応
    ssl._create_default_https_context = ssl._create_unverified_context
    print(f"Tenforflow Version : {tf.version.VERSION}")

    # 入力データのフォーマット
    trainds = tfds.load("coco_captions", split="train", data_dir=GCS_DIR)
    trainds = trainds.map(
        get_image_label,
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE)
    trainds = trainds.prefetch(buffer_size=tf.data.AUTOTUNE)

    f, ax = plt.subplots(1, 4, figsize=(20, 5))
    for idx, data in enumerate(trainds.take(4)):
        ax[idx].imshow(data["image_tensor"].numpy())
        caption = "\n".join(wrap(data["caption"].numpy().decode("utf-8"), 30))
        ax[idx].set_title(caption)
        ax[idx].axis("off")
    plt.savefig("image_captions.png")
    print("Complete Create Image")
    
    # start, endトークンの付与
    trainds = trainds.map(add_start_end_token)

    tokenizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        standardize=standardize,
        output_sequence_length=MAX_CAPTION_LEN
    )
    tokenizer.adapt(trainds.map(lambda x: x["caption"]))

    # サンプル文字列のトークン化
    tokenizer(["<start>this is a sentence <end>"])
    sample_captions = []
    for d in trainds.take(5):
        sample_captions.append(d["caption"].numpy())
    print(f"Sample Captions : {sample_captions}\n")
    print(f"Token of Sample Caption : {tokenizer(sample_captions)}\n")
    print("SAMPLE:")
    for wordid in tokenizer([sample_captions[0]])[0]:
        print(f'{tokenizer.get_vocabulary()[wordid]}', end=" ")

    # 文字列 -> トークン 変換関数
    word_to_token = StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary()
    ) 
    # トークン -> 文字列 変換関数
    token_to_word = StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True
    )

    BATCH_SIZE = 32
    batch_ds = (
        trainds.map(create_ds_fn(tokenizer))
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    for (img, caption), label in batch_ds.take(2):
        print(f"Image shape    : {img.shape}")
        print(f"Caption ashape : {caption.shape}")
        print(f"Label shape    : {label.shape}")
        print(f"caption[0]     : {caption[0]}")
        print(f"label[0])      : {label[0]}")
    
    
    # イメージデータのベクトル化
    FEATURE_EXTRACTOR.trainable = False
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    image_feature = FEATURE_EXTRACTOR(image_input)
    x = Reshape((FEATURES_SHAPE[0] * FEATURES_SHAPE[1], FEATURES_SHAPE[2]))(
        image_feature
    )
    encoder_output = Dense(ATTENTION_DIM, activation='relu')(x)
    encoder = tf.keras.Model(input=image_input, outputs=encoder_output)
    encoder.summary()

    # decoder
    word_intput = Input(shape=(MAX_CAPTION_LEN,), name="words")
    embed_x = Embedding(VOCAB_SIZE, ATTENTION_DIM)(word_intput)
    # 各レイヤーの作成
    # GRU
    decoder_gru = GRU(
        ATTENTION_DIM,
        retrurn_sequences=True,
        return_state=True
    )
    gru_output, gru_state = decoder_gru(embed_x)
    # Attention
    decoder_attention = Attention()
    context_vector = decoder_attention([gru_output, encoder_output])
    # Add
    addition= Add([gru_output, context_vector])
    # 正規化
    layer_norm = LayerNormalization(axis=1)
    layer_norm_out = layer_norm(addition)
    # Dense
    decoder_output_dense = Dense(VOCAB_SIZE)
    decoder_output = decoder_output_dense(layer_norm_out)

    decoder = tf.keras.Model(
        input=[word_intput, encoder_output], 
        outputs=decoder_output
    )
    tf.keras.utils.plot_model(decoder)
    decoder.summary()

    # Training
    image_caption_train_model = tf.keras.Model(
        input=[image_input, word_intput],
        outputs=decoder_output
    )
    image_caption_train_model.compile(
        optimizer='adam',
        loss_function=loss_function
    )
    # TrainingLoop
    history = image_caption_train_model.fit(
        batch_ds,
        epochs=1
    )

    # トレーニング済みGRUを再利用。ただし結果受信した場合は更新
    gru_state_input = Input(shape=(ATTENTION_DIM,), name='gru_state_input')
    gru_output, gru_state = decoder_gru(embed_x, initial_state=gru_state_input)
    # その他レイヤーも再利用
    context_vector = decoder_attention([gru_output, encoder_output])
    addition_output = Add()([gru_output, context_vector])
    layer_norm_output = layer_norm(addition_output)
    decoder_output = decoder_output_dense(layer_norm_output)

    decoder_pred_model = tf.keras.Model(
        inputs=[word_intput, gru_state_input, encoder_output],
        outputs=[decoder_output, gru_state]
    )

    MINIMUM_SENTENCE_LENGTH = 5
    file_path = get_image_path("baseball.jpeg")
    for i in range(5):
        image, caption = predict_caption(encoder, decoder_pred_model, word_to_token, tokenizer, file_path)
        print(" ".join(caption[:-1]) + ".")
    img = tf.image.decode_jpeg(tf.io.read_file(file_path), channels=IMG_CHANNEL)
    plt.imshow(img)
    plt.axis=False



def get_image_label(example):
    print(f"Data : {example}")
    caption = example["captions"]["text"][0] # イメージ毎の最初のキャプション
    img = example["image"]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255
    return {"image_tensor" : img, "caption" : caption}

def add_start_end_token(data):
    start = tf.convert_to_tensor("<start>")
    end = tf.convert_to_tensor("<end>")
    data["caption"] = tf.strings.join(
        [start, data["caption"], end],
        separator=" "
    )
    return data

def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(
        inputs,
        r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~]?", # 記号
        ""
    )

def create_ds_fn(tokenizer):
    def fn(data):
        img_tensor = data["image_tensor"]
        caption = tokenizer(data["caption"])

        target = tf.roll(caption, -1, 0)
        zeros = tf.zeros([1], dtype=tf.int64)
        target = tf.concat((target[:-1], zeros), axis=-1)
        return (img_tensor, caption), target
    return fn

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    loss_ = loss_object(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    sentence_len = tf.reduce_sum(mask)
    loss_ = loss_[:sentence_len]
    return tf.reduce_mean(loss_, 1)


def predict_caption(encoder, decoder_pred_model, word_to_token, tokenizer, filename):
    gru_state = tf.zeros((1, ATTENTION_DIM))

    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=IMG_CHANNEL)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255

    features = encoder(tf.expand_dims(img, axis=0))
    dec_input = tf.expand_dims([word_to_token("<start>")], 1)
    result = []
    for i in range(MAX_CAPTION_LEN):
        predictions, gru_state = decoder_pred_model(
            [dec_input, gru_state, features]
        )

        top_probs, top_idxs = tf.math.top_k(
            input=predictions[0][0],
            k=10,
            sorted=False
        )
        chosen_id = tf.random.categorical([top_probs], 1)[0].numpy()
        predicted_id = top_idxs.numpy()[chosen_id][0]

        result.append(tokenizer.get_vocabulary()[predicted_id])

        if predicted_id == word_to_token("<end>"):
            return img, result
        
        dec_input = tf.expand_dims([predicted_id], 1)
        return img, result

def get_image_path(file_name):
    image_path = os.path.join(os.path.expanduser("~"),
                    "work",
                    "generative-ai-for-developers-learning-path",
                    "image",
                    file_name)
    return image_path 