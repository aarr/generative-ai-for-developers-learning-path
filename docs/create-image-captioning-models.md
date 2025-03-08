# [Create Image Captioning Models](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/multi_modal/solutions/image_captioning.ipynb)

## Local環境で実行

* [Apple Developer Site](https://developer.apple.com/metal/tensorflow-plugin/)の通り実行

  tensorFlow バージョン 2.13 以降の場合は、pipで直接インストール可能

  ```bash
  python -m pip install tensorflow
  ```

* 実行時にエラー

  * TensorFlowに関わるライブラリのモジュールバージョンを合わせる
    結果下記で動作した

    ```bash
    python -m pip list | grep tenforflow
    ```

    ```text
    tensorflow                    2.18.0
    tensorflow-datasets           4.9.3
    tensorflow-estimator          2.15.0
    tensorflow-hub                0.16.1
    tensorflow-io-gcs-filesystem  0.37.1
    tensorflow-macos              2.16.2
    tensorflow-metadata           1.16.1
    tensorflow-metal              1.2.0
    tensorflow-model-optimization 0.8.0
    tensorflow-text               2.18.0
    ```

* その他

  * 色々とライブラリのバージョンが合わない

    ```bash
    python -m pip check
    ```

    インストール可能なバージョン検索
    （indexはその内廃止になるかも）

    ```bash
    python -m pip index versions library
    ```
