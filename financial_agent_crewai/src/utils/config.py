from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# User query
USER_QUERY = 'What are the main points discussed in the Google 10-K filing for 2024?'
# USER_QUERY = 'What are the differences in product strategy between Apple and Google filings in 2024?'

# Comparison query
COMPARISON_QUERY = (
    'Please write a conclusion/summary for the provided context, '
    f'specifically focusing around the user query: {USER_QUERY}'
)

MAX_NEWS_PER_TICKER = 2

SOURCE_GENERIC_SEARCH = True
SOURCE_SEC_FILINGS = False
SOURCE_YFINANCE_NEWS = False
SOURCE_YFINANCE_STOCK = False

DECOMPOSITION_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
GENERIC_RESEARCH_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
INFORMATION_EXTRACTION_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
RAG_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
REPORT_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
SEC_EDGAR_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
YFINANCE_NEWS_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
YFINANCE_STOCKS_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'

TEMPERATURE = 0
