"""学習用のスクリプトです。"""
from src.recommend.ml.imf.fetcher import fetch_data_for_train
from src.recommend.ml.imf.trainer import Trainer

if __name__ == "__main__":
    raw_df = fetch_data_for_train()
    trainer = Trainer().execute(raw_df=raw_df)
