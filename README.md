# Serendipity ML Server

## プロジェクトの責務

Serendipity における機械学習アルゴリズムの API を提供します。

## アーキテクチャの概要

本プロジェクトは Serverless Framework によりデプロイされます。AWS Lambda のアップロード容量制限をクリアするため、Docker イメージを焼いてアップロードします。手元の python の仮想環境は pyenv を用いて作成します。また、依存関係は poetry により管理されます。

## ローカル環境での実行の仕方

Uvicorn を使用することにより仮想サーバーを立てることができます。以下の手順に従ってください。

1. `poetry shell` により `pyenv` の仮想環境に入る
2. `uvicorn src.main:app` と打ち込むことで仮想サーバーを立てる

## デプロイの方法

`sls deploy` によりデプロイをすることができます。ただし、`poetry export --without-hashes -f requirements.txt --output requirements.txt` と打ち込むことであらかじめ requirement.txt を作成しておく必要があります。

```
failed to solve with frontend dockerfile.v0: failed to create LLB definition: unexpected status code [manifests 3.8]: 403 Forbidden
```

とエラーが出る場合があります。この場合は以下のコマンドを実行することによりまずは認証を得てください。

```
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
```

## 機械学習モデルの実験管理

mlflow を用いて実験管理を行っています。レコメンドエンジン (`recommend`) の部分を使ってその使い方を説明しておきます。参考 URL は主に以下です。

https://qiita.com/c60evaporator/items/e0eb1a0c521d1310d95d

### 具体的な使い方

`src/recommend/ml/mlflow/experiment.py` を実行することにより、パラメータ及び評価指標を sqlite データベースに保存できます。パラメータなどは `src/recommend/ml/mlflow/config.py` にて `parameter`, `EXPERIMENT_NAME` を指定し、レポジトリ直下で以下のコードを実行すれば学習及びパラメータ・評価指標保存のプロセスが走ります。

```
python3 -m src.recommend.ml.mlflow.experiment
```

結果表示のための UI を出力するためには以下の手順で実行します。

1. まずは db が保存されているディレクトリに移動します。ここでは `src/recommend/ml/mlflow/sqlite/` です。
2. このディレクトリに移動したらコンソール上で以下のコマンドを打ち込みます。

```
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
