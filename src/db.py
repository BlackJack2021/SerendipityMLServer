from src.db_config import config
from src.secret import rds_secret
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

async_driver = config["async_driver"]
sync_driver = config["sync_driver"]
user = rds_secret["user"]
password = rds_secret["password"]
endpoint = config["endpoint"]
port = config["port"]
database = config["database"]
charset = config["charset"]

# 非同期通信用の設定
ASYNC_DB_URL = (
    f"{async_driver}://{user}:{password}@{endpoint}:{port}/{database}?charset={charset}"
)
async_engine = create_async_engine(ASYNC_DB_URL, echo=False)
async_session = sessionmaker(
    autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession
)
Base = declarative_base()


async def get_db():
    async with async_session() as session:
        yield session


# 同期通信用の設定
SYNC_DB_URL = (
    f"{sync_driver}://{user}:{password}@{endpoint}:{port}/{database}?charset={charset}"
)
engine = create_engine(SYNC_DB_URL)
sync_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
