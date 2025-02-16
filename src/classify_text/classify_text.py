import os
import warnings

import shutil
import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from google.cloud import aiplatform
from official.nlp import optimization
import ssl

warnings.filterwarnings("ignore")
os.environ[ "TF_CPP_MIN_LOG_LEVEL" ] = "2"
# Tensorflow-hubはTensorflow2.16と互換性がない
# https://github.com/tensorflow/hub/issues/903
# os.environ['TF_USE_LEGACY_KERAS']='1'

tf.get_logger().setLevel("ERROR")

tfhub_handle_encoder = ( 
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    )
tfhub_handle_preprocess = ( 
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )

# メイン処理
def main():
    print("START")
    print("Num Gpus Available: ", len(tf.config.list_physical_devices("GPU")))

    # ファイルダウンロード
    work_base_dir = get_work_dir()
    print(f"work_base_dir : {work_base_dir}")
    download_dataset(work_base_dir)
    train_dir = get_train_dir(work_base_dir)
    print(f"train_dir : {train_dir}")

    AUROTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    row_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=seed
    )

    class_names = row_train_ds.class_names
    train_ds = row_train_ds.cache().prefetch(buffer_size=AUROTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed
    )
    val_ds = val_ds.cache().prefetch(buffer_size=AUROTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        get_test_dir(work_base_dir),
        batch_size=batch_size
    )
    test_ds = test_ds.cache().prefetch(buffer_size=AUROTUNE)

    
    # いくつか中身を確認
    for text_batch, label_batch in train_ds.take(1):
        for i in range(3):
            print(f"Review : {text_batch.numpy()[i]}")
            label = label_batch.numpy()[i]
            print(f"Label : {label} {class_names[label]}\n")

    
    print(f"BERT model selected            : {tfhub_handle_encoder}")
    print(f"BERT preprocesse auto selected : {tfhub_handle_preprocess}")

    # 前処理を設定
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    text_test = ["this is such an amazing movie!"]
    text_preprocessed = bert_preprocess_model(text_test)
    # 前処理のキー
    print(f"Key        : {list(text_preprocessed.keys())}")
    # input_word_idsはトークン化された単語のID
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    # input_maskはマスキングしているトークン
    # BERTにおいて、モデルトレーニングのタスク：MLM(Masked language modeling)を行う際に、
    # 15%の文字列をマスクして学習させる。その為の前処理と思われる
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    # input_type_idsは入力文のID
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}\n')

    # Encoder実行
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    bert_result = bert_model(text_preprocessed)
    print(f"Loaded BERT            : {tfhub_handle_encoder}")
    print(f'Pooled Outputs Shape   : {bert_result["pooled_output"].shape}')
    print(f'Pooled Outputs Value   : {bert_result["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape : {bert_result["sequence_output"].shape}')
    print(f'Sequence Outputs Value : {bert_result["sequence_output"][0, :12]}\n')

    # BuildModel
    dropout_rate = 0.15
    classifier_model = build_classifier_model(dropout_rate)
    bert_raw_result = classifier_model(tf.constant(text_test))
    print(f"BERT Result : {bert_raw_result}")
    # Localにイメージファイル出力(model.png)
    tf.keras.utils.plot_model(classifier_model)

    # 損失関数
    loss = tf.keras.losses.BinaryCrossentropy() # クロスエントロピー誤差
    metrics = tf.metrics.BinaryAccuracy()       # 精度

    epochs = 1
    steps_per_epochs = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epochs * epochs
    num_warmup_steps = int(0.1 * num_train_steps) 
    
    init_lr = 3e-5   # 学習率
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw"      # 重み更新ロジック
    )
    classifier_model.compile(optimizer, loss, metrics)
    print(f"Training model with : {tfhub_handle_encoder}")
    history = classifier_model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    loss = accuracy = classifier_model.evaluate(test_ds)
    print(f"Loss     : {loss}")
    print(f"Accuracy : {accuracy}")

    history_dict = history.history
    print(f"History Key : {history_dict.keys()}")
    acc = history_dict["binary_accuracy"]
    val_acc = history_dict["val_binary_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label= "Validation Loss")
    plt.title("Training and Validation Loss")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, "r", label="Traingin acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")

    # Modelの保存
    dataset_name = "imdb"
    saved_model_path = "./{}_bert".format(dataset_name.replace("/", "_"))
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    EXPORT_PATH = os.path.join(saved_model_path, TIMESTAMP)
    classifier_model.save(EXPORT_PATH, include_optimizer=False)

    # Modelのロード
    reloaded_model = tf.saved_model.load(EXPORT_PATH)

    examples = [
        "this is such an amazing movie!",
        "The movie was great!",
        "The movie was meh.",
        "The movie was okish.",
        "The movie was terrible..."
    ]
    reloaded_result = reloaded_model(tf.constant(examples))
    original_result = classifier_model(tf.constant(examples))
    print("Result from saved model :")
    print_my_example(examples, reloaded_result)
    print("Result from the model in memory : ")
    print_my_example(examples, original_result)

    print("END")


# データセットダウンロード
def download_dataset(work_base_dir):
    # 証明書無効化
    # 無効化しないとエラーになるので暫定対応
    ssl._create_default_https_context = ssl._create_unverified_context

    file_name = "aclImdb_v1.tar.gz"
    url = "https://ai.stanford.edu/~amaas/data/sentiment/" + file_name
    dataset = tf.keras.utils.get_file(
        file_name,
        url,
        untar = True,
        cache_dir = work_base_dir,
        cache_subdir = ""
    )

    train_dir = get_train_dir(work_base_dir)
    remove_dir = os.path.join(train_dir, "unsup")
    # 対象ディレクトリを丸ごと削除
    shutil.rmtree(remove_dir, ignore_errors=True)

# ディレクトリ作成
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ワークディレクトリ取得
def get_work_dir():
    work_base_dir = os.path.join(os.path.expanduser("~"),
                    "work",
                    "generative-ai-for-developers-learning-path")
    make_dir(work_base_dir)
    return work_base_dir

# データセットディレクトリ取得
def get_dataset_dir(work_base_dir):
    base_dir = os.path.join(work_base_dir, "aclImdb")
    make_dir(base_dir)
    return base_dir

# トレーニングディレクトリ取得
def get_train_dir(work_base_dir):
    train_dir = os.path.join(get_dataset_dir(work_base_dir), "train")
    make_dir(train_dir)
    return train_dir

# テストディレクトリ取得
def get_test_dir(work_base_dir):
    test_dir = os.path.join(get_dataset_dir(work_base_dir), "test")
    make_dir(test_dir)
    return test_dir

# BuildModel
def build_classifier_model(dropout_rate=0.1):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")

    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(dropout_rate)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)
    return tf.keras.Model(text_input, net)

def print_my_example(inputs, results):
    result_for_printing = [
        f"input : {inputs[i]:<30} : score : {results[i][0]:.6f}"
        for i in range(len(inputs))
    ]
    print(*result_for_printing, sep="\n")
    print()