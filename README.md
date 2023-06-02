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

`sls deploy` によりデプロイをすることができます。ただし、`poetry export --without-has
hes -f requirements.txt --output requirements.txt` と打ち込むことであらかじめ requirement.txt を作成しておく必要があります。

```
failed to solve with frontend dockerfile.v0: failed to create LLB definition: unexpected status code [manifests 3.8]: 403 Forbidden
```

とエラーが出る場合があります。この場合は以下のコマンドを実行することによりまずは認証を得てください。

```
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
```
