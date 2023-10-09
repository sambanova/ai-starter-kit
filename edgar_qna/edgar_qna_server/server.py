import os

from edgar_sec_qa_hosted import SecFilingQa
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app.sec_qa = None


class Config(BaseModel):
    ticker: str


class Query(BaseModel):
    ticker: str
    query: str


def load_edgar_data(ticker):
    config = {}
    config["ticker"] = ticker
    config["vector_db_url"] = os.getenv("VECTOR_DB_URL")
    sec_qa = SecFilingQa(config=config)
    sec_qa.init_embeddings()
    sec_qa.init_models()
    sec_qa.vector_db_sec_docs()
    sec_qa.retreival_qa_chain(ticker)
    return sec_qa


@app.post("/ask_llm")
async def ask_llm(query: Query):
    sec_qa = load_edgar_data(query.ticker)
    result = sec_qa.answer_sec(query.query)
    return {"message": result}
