import sys

sys.path.append('../')
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from edgar_qna.src.edgar_sec import SecFiling

PERSIST_DIRECTORY = '../data/vectordbs'

load_dotenv('../../export.env')

app = FastAPI()
assert hasattr(app, 'sec_qa')
app.sec_qa = None


class Config(BaseModel):
    ticker: str


class Query(BaseModel):
    ticker: str
    query: str


def set_retrieval_qa_chain(ticker: Any) -> Any:
    config = {}
    config['ticker'] = ticker
    config['persist_directory'] = f'{PERSIST_DIRECTORY}/{ticker}'
    sec_qa = SecFiling(config)
    sec_qa.init_llm_model()
    sec_qa.create_load_vector_store()
    sec_qa.retrieval_qa_chain()
    return sec_qa


@app.post('/ask_llm')
async def ask_llm(query: Query) -> Dict[str, Any]:
    sec_qa = set_retrieval_qa_chain(query.ticker)
    response = sec_qa.answer_sec(query.query)
    return {'message': response['answer']}
