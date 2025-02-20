import sys
import logging
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from financial_agent_crewai.queue import StreamToQueue
from financial_agent_crewai import schemas
from financial_agent_crewai.main import FinancialFlow
from financial_agent_crewai.src.financial_agent_crewai.config import *
from financial_agent_crewai.utils.utilities import *


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


@app.post('/agent/predict', response_model=schemas.AgentFinalOutput)
def financial_agent(user_input: schemas.UserInput) -> schemas.AgentFinalOutput:
    try:        
        logger.info(f"Received request for financial prediction with input: {user_input}")
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
        logger.error(f"Error occurred during financial prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred")


@app.post('/agent/stream')
async def financial_agent_stream(user_input: schemas.UserInput) -> StreamingResponse:
    queue = Queue()

    def run_financial_flow() -> None:
        """Run the financial flow while capturing logs."""
        try:
            logger.info(f"Received request for financial prediction stream with input: {user_input}")

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
            logger.error(f"Error during financial prediction stream: {str(e)}")
            sys.stderr.write(f"Error during financial flow: {str(e)}\n")
        finally:
            sys.stdout = sys.__stdout__  # Restore stdout
            queue.put(None)  # Signal end of logs

    try:
        thread = Thread(target=run_financial_flow, daemon=True)
        thread.start()

        return StreamingResponse(log_stream(queue), media_type='text/plain')

    except Exception as e:
        logger.error(f"An error occurred while starting the stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while starting the stream")
