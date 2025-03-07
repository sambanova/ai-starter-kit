import io
import json
import logging
import os
import sys
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Annotated, Dict

import redis
from cryptography.fernet import InvalidToken
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from financial_agent_crewai.src.exceptions import APIKeyNotFoundError, InvalidSessionError
from financial_agent_crewai.src.financial_agent_crewai.config import *
from financial_agent_crewai.src.main import FinancialFlow
from financial_agent_crewai.utils.utilities import log_stream, schedule_temp_dir_deletion
from financial_agent_crewai.web_app.backend import oauth2, schemas
from financial_agent_crewai.web_app.backend.config import settings
from financial_agent_crewai.web_app.backend.session.credentials_manager import APIKeyManager
from financial_agent_crewai.web_app.backend.session.user_manager import UserSessionManager
from financial_agent_crewai.web_app.backend.streaming_queue import StreamToQueue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=0)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

sambanova_api_key_header = APIKeyHeader(name='x-sambanova-key', scheme_name='sambanova_key')
serper_api_key_header = APIKeyHeader(name='x-serper-key', scheme_name='serper_key')
key_manager = APIKeyManager()


# Working directories
current_dir = Path(__file__).resolve().parent
kit_dir = current_dir.parent

# Exceptions
session_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail='Invalid or expired session',
)
credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail='Could not validate credentials',
)


@app.post('/keys')
async def store_keys(
    response: Response,
    sambanova_api_key: str = Depends(sambanova_api_key_header),
    serper_api_key: str = Depends(serper_api_key_header),
) -> Dict[str, str]:
    api_keys = {'sambanova_api_key': sambanova_api_key, 'serper_api_key': serper_api_key}

    session_token = UserSessionManager.create_session(api_keys, key_manager, redis_client)
    access_token = oauth2.create_access_token({'session_token': session_token})

    response.set_cookie(
        key='access_token',
        value=access_token,
        httponly=True,
        secure=True,
        samesite='strict',
        max_age=7200,
    )

    return {'access_token': access_token, 'token_type': 'bearer'}


@app.post('/agent/predict', response_model=schemas.AgentFinalOutput)
def financial_agent(user_input: schemas.UserInput) -> schemas.AgentFinalOutput:
    """
    Handles a POST request to predict financial agent outcomes based on user input.

    Args:
        user_input: The user input containing query and data sources.

    Returns:
        A response containing the prediction results.

    Raises:
        HTTPException: If an error occurs during the prediction process.
    """
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
async def financial_agent_stream(
    user_input: schemas.UserInput, access_token: Annotated[str | None, Cookie()] = None
) -> StreamingResponse:
    """
    Handles a POST request to stream financial agent prediction results based on user input.

    Args:
        user_input: The user input containing query and data sources.

    Returns:
        StreamingResponse: A response streaming the logs generated during the financial prediction.

    Raises:
        HTTPException: If an error occurs while starting the stream.
    """
    if access_token is None:
        raise session_exception

    # Extract session token
    session_token = oauth2.verify_access_token(access_token, credentials_exception)

    # Create the cache path
    cache_path = create_cache_path_session_token(kit_dir, session_token)
    Path(cache_path).mkdir(exist_ok=True)

    # Schedule cache deletion after 30 minutes
    schedule_temp_dir_deletion(str(cache_path), 30)

    if session_token.session_token is None:
        raise session_exception
    try:
        api_keys = UserSessionManager.get_session_keys(session_token.session_token, key_manager, redis_client)

    except InvalidSessionError:
        raise session_exception
    except InvalidToken:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')

    queue: Queue[str] = Queue()

    try:
        financial_flow = FinancialFlow(
            query=user_input.user_query,
            source_generic_search=user_input.source_generic_search,
            source_sec_filings=user_input.source_sec_filings,
            source_yfinance_news=user_input.source_yfinance_news,
            source_yfinance_stocks=user_input.source_yfinance_stocks,
            cache_path=cache_path,
            sambanova_api_key=api_keys.get('sambanova_api_key'),
            serper_api_key=api_keys.get('serper_api_key'),
        )
    except APIKeyNotFoundError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f'No api key found')

    def run_financial_flow() -> None:
        """
        Executes the financial prediction flow and streams logs to the queue.
        Logs are captured and sent to the response stream.

        Raises:
            Exception: If an error occurs during the financial prediction flow.
        """
        try:
            # Redirect logs to queue
            sys.stdout = StreamToQueue(queue)
            financial_flow.kickoff()
        except Exception as e:
            logger.error(f'Error during agent flow: {str(e)}')
            queue.put(json.dumps({'type': 'error', 'content': 'internal error during agent flow'}))
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
            # Signal end of logs
            queue.put(None)  # type: ignore

    try:
        thread = Thread(target=run_financial_flow, daemon=True)
        thread.start()

        return StreamingResponse(log_stream(queue), media_type='text/plain')

    except Exception as e:
        logger.error(f'An error occurred while starting the stream: {str(e)}')
        raise HTTPException(status_code=500, detail=f'An error occurred while starting the stream')


@app.get('/report/md')
async def get_report_md(access_token: Annotated[str | None, Cookie()] = None) -> StreamingResponse:
    """
    Retrieves and streams a Markdown report file to the client.

    Returns:
        A dictionary containing the streaming response of the Markdown file.

    Raises:
        HTTPException: If the Markdown file does not exist.
    """
    if access_token is None:
        raise session_exception

    # Extract session token
    session_token = oauth2.verify_access_token(access_token, credentials_exception)

    # Create the cache path
    cache_path = create_cache_path_session_token(kit_dir, session_token)

    # Create the final report Markdown path
    md_file_path = os.path.join(cache_path, 'report.md')

    if not os.path.exists(md_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Markdown file not found')

    with open(md_file_path, 'r') as f:
        md_data = f.read()

    return StreamingResponse(
        io.BytesIO(md_data.encode()),
        media_type='text/markdown',
        headers={'Content-Disposition': 'attachment; filename=report.md'},
    )


@app.get('/report/pdf')
async def get_report_pdf(access_token: Annotated[str | None, Cookie()] = None) -> StreamingResponse:
    """
    Retrieves and streams a PDF report file to the client.

    Returns:
        A dictionary containing the streaming response of the PDF file.

    Raises:
        HTTPException: If the PDF file does not exist.
    """
    if access_token is None:
        raise session_exception

    # Extract session token
    session_token = oauth2.verify_access_token(access_token, credentials_exception)

    # Create the cache path
    cache_path = create_cache_path_session_token(kit_dir, session_token)

    # Create the final report PDF path
    pdf_file_path = os.path.join(cache_path, 'report.pdf')

    if not os.path.exists(pdf_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='PDF file not found')

    with open(pdf_file_path, 'rb') as f:
        pdf_data = f.read()

    return StreamingResponse(
        io.BytesIO(pdf_data),
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=report.pdf'},
    )


def create_cache_path_session_token(kit_dir: Path, session_token: str) -> Path:
    """Create the cache path from user token."""

    return kit_dir.parent.parent / 'scratch' / 'financial_assistant' / 'cache' / f'cache_{session_token}'
