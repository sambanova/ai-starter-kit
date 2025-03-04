import io
import json
import logging
import sys
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import schemas
from src.financial_agent_crewai.config import *
from src.main import FinancialFlow
from .streaming_queue import StreamToQueue
from utils.utilities import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

pdf_file_path = 'cache/report.pdf'
md_file_path = 'cache/report.md'


@app.post('/agent/predict', response_model=schemas.AgentFinalOutput)
def financial_agent(user_input: schemas.UserInput) -> schemas.AgentFinalOutput:
    try:
        logger.info(f'Received request for financial prediction with input: {user_input}')
        financial_flow = FinancialFlow(
            query=user_input.user_query,
            source_generic_search=user_input.source_generic_search,
            source_sec_filings=user_input.source_sec_filings,
            source_yfinance_news=user_input.source_yfinance_news,
            source_yfinance_stocks=user_input.source_yfinance_stocks,
        )
        results = financial_flow.kickoff()

        response = schemas.AgentFinalOutput(**results)
        return response

    except Exception as e:
        logger.error(f'Error occurred during financial prediction: {str(e)}')
        raise HTTPException(status_code=500, detail=f'An error occurred')


@app.post('/agent/stream')
async def financial_agent_stream(user_input: schemas.UserInput) -> StreamingResponse:
    queue = Queue()

    def run_financial_flow() -> None:
        """Run the financial flow while capturing logs."""
        try:
            logger.info(f'Received request for financial prediction stream with input: {user_input}')

            sys.stdout = StreamToQueue(queue)  # Redirect logs to queue

            financial_flow = FinancialFlow(
                query=user_input.user_query,
                source_generic_search=user_input.source_generic_search,
                source_sec_filings=user_input.source_sec_filings,
                source_yfinance_news=user_input.source_yfinance_news,
                source_yfinance_stocks=user_input.source_yfinance_stocks,
            )
            financial_flow.kickoff()
        except Exception as e:
            logger.error(f'Error during agent flow: {str(e)}')
            queue.put(json.dumps({'type': 'error', 'content': 'internal error during agent flow'}))
        finally:
            sys.stdout = sys.__stdout__  # Restore stdout
            queue.put(None)  # Signal end of logs

    try:
        thread = Thread(target=run_financial_flow, daemon=True)
        thread.start()

        return StreamingResponse(log_stream(queue), media_type='text/plain')

    except Exception as e:
        logger.error(f'An error occurred while starting the stream: {str(e)}')
        raise HTTPException(status_code=500, detail=f'An error occurred while starting the stream')


@app.get('/report/pdf')
async def get_report() -> dict:
    if not os.path.exists(pdf_file_path):
        raise HTTPException(status_code=404, detail='PDF file not found')

    with open(pdf_file_path, 'rb') as f:
        pdf_data = f.read()

    return StreamingResponse(
        io.BytesIO(pdf_data),
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=report.pdf'},
    )


@app.get('/report/md')
async def get_report() -> dict:
    if not os.path.exists(md_file_path):
        raise HTTPException(status_code=404, detail='Markdown file not found')

    with open(md_file_path, 'r') as f:
        md_data = f.read()

    return StreamingResponse(
        io.BytesIO(md_data.encode()),
        media_type='text/markdown',
        headers={'Content-Disposition': 'attachment; filename=report.md'},
    )
