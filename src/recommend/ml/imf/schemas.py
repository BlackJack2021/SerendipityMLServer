from pandera import Column, DataFrameSchema
from pandera.dtypes import Timestamp

# 学習に利用する生データのスキーマ
raw_data_for_train_schema = DataFrameSchema(
    columns={
        "user_id": Column(str),
        "datetime": Column(Timestamp),
        "edinet_code": Column(str),
        "filer_name": Column(str),
    }
)
