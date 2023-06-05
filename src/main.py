from fastapi import FastAPI
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware

from src.api_documents import docs_url, redoc_url
from src.secret import api_server_origins
import src.recommend.routers as recommend

app = FastAPI(docs_url=docs_url, redoc_url=redoc_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_server_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")
def hello():
    return {"message": "hello world!"}


app.include_router(recommend.router)
handler = Mangum(app)
