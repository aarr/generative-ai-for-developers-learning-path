# [Encoder-Decoder Architecture Learning](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/text_models/solutions/text_generation.ipynb)

## LOCAL実行

* Local環境でTenforflowを動かしてモデル作成、計算結果出力

  ```bash
  # conda環境作成（python 3.9）
  conda create --prefix ./env39 python=3.9
  conda環境作成（python 3.8）
  # conda create --prefix ./env python=3.8
  # conda環境有効化（プロジェクトルートディレクトリ）
  # conda activate ./env
  # 実行
  python src/text_generation
  ```

  不要な環境は削除

  ```bash
  conda env remove --prefix env
  ```
