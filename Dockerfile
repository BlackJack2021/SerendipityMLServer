FROM public.ecr.aws/lambda/python:3.8

COPY ./requirements.txt /opt/

# pyproject.toml ファイルより、必要な依存関係をインストールする
RUN pip install --upgrade pip \
    && pip install -r /opt/requirements.txt


# デプロイ時のエラーを修正するために追加
RUN yum install -y libgomp

COPY ./src ${LAMBDA_TASK_ROOT}/src
CMD ["src.main.handler"]

