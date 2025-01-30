# [Generative AI for Developers Learning Path](https://www.cloudskillsboost.google/paths/183?locale=ja)

## Python環境構築

* プロジェクト作成
  * APIキー発行（GCP）

  * [hatch](https://hatch.pypa.io/latest/install/)
    * 機能
      * backend build
      * Environment management
    * Project作成

      ```bash
      hatch new text-generation
      ```

  * TensorflowのBuild
    * M4CPUには未対応。そのため、自分でソースからビルド
      `The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.`

    * tensorflowのソース取得

      ```bash
      git clone https://github.com/tensorflow/tensorflow.git
      ```

    * bazelをインストール
      * Homebrewでインストールできない場合は、[こちら](https://github.com/bazelbuild/bazel/releases)からダウンロード
      * bazel-6.5.0-installer-darwin-x86_64.sh

      ```bash
      # バージョンは6.5.0
      brew install bazel
      ```

    * [basel](https://www.tensorflow.org/install/source?hl=ja#macos)でビルド
      * macはGPU対応されていないので、CPU対応でビルド

        ```bash
        bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
        ```

      * ビルドオプションをや待ったのか、環境に合っていないとエラー

        ```text
        The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.
        ```

    * Compile済みのwheelファイルを[ダウンロード](https://www.tensorflow.org/install/pip?hl=ja#macos)
      * Python3.11の[wheelファイル](https://storage.googleapis.com/tensorflow/versions/2.16.2/tensorflow-2.16.2-cp311-cp311-macosx_10_15_x86_64.whl)

      * pipでインストール

        ```bash
        pip install tensorflow-2.16.2-cp311-cp311-macosx_10_15_x86_64.whl
        ```

      * 再びエラー

    * conda経由で[インストール](https://qiita.com/xtrizeShino/items/56e3c1e027107debe331)
      * condaで環境の作成、Activate

        ```bash
        # 作業ディレクトリを作成
        $ conda create --prefix ./env python=3.8
        # Python3.8ベースの仮想環境「env」を作成
        $ conda activate ./env
        ```

      * tensorflow系パッケージをinsall

        ```bash
        conda install -c apple tensorflow-deps
        python -m pip install tensorflow-macos
        python -m pip install tensorflow-metal
        conda install jupyter pandas numpy matplotlib scikit-learn
        ```

    * GPUアクセス確認

      ```python
      # Check for TensorFlow GPU access
      print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

      # See TensorFlow version
      print(f"TensorFlow version: {tf.__version__}")
      ```

      結果
      `TensorFlow has access to the following devices:`
      `[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
      `TensorFlow version: 2.13.0`

## Learning Path

1. [Encoder-Decoder Architecture Learning](./docs/encoder-decoder-architecture-learning.md)

1. [Transformer Models and BERT Model](./docs/transformer-models-and-bert-model.md)
