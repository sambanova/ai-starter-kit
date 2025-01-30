from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# User query
# USER_QUERY = 'What is the product strategy of Google as per its 10-K filing in 2024?'
USER_QUERY = 'What are the differences in product strategy between Apple and Google filings in 2024?'

# Comparison query
COMPARISON_QUERY = (
    'Please write a conclusion/summary for the provided context, '
    f'specifically focusing around the user query: {USER_QUERY}.\n'
)

# Maximum number of urls in generic Google web search or Yahoo Finance web search
MAX_NEWS = 5
# Maximum news per ticker symbol
MAX_NEWS_PER_TICKER = 2

# Data sources
SOURCE_GENERIC_SEARCH = False
SOURCE_SEC_FILINGS = True
SOURCE_YFINANCE_NEWS = False
SOURCE_YFINANCE_STOCK = False

# LLMs by crew
GENERAL_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'

CONTEXT_ANALYSIS_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
DECOMPOSITION_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
GENERIC_RESEARCH_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
INFORMATION_EXTRACTION_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
RAG_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
QA_MODEL = 'Meta-Llama-3.1-70B-Instruct'
REPORT_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
SEC_EDGAR_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
YFINANCE_NEWS_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'
YFINANCE_STOCKS_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'

# LLM temperature
TEMPERATURE = 0
