from pathlib import Path

# Level of verbosity
VERBOSE = False

# Specify the directory and file path
CACHE_DIR = Path('cache/')
YFINANCE_STOCKS_DIR = CACHE_DIR / 'yfinance_stocks'

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# Data sources
SOURCE_GENERIC_SEARCH = False
SOURCE_SEC_FILINGS = False
SOURCE_YFINANCE_NEWS = False
SOURCE_YFINANCE_STOCK = True

# User query
USER_QUERY = 'Please give me an analysis of Google liabilities and assets for the last 6 months.'

# Comparison query
COMPARISON_QUERY = (
    'Please compare the provided contexts, focusing specifically on the user query: '
    f'{USER_QUERY}.\n'
    f'Do not exceed {MAX_SECTION_WORDS/2} words.'
)
# Number of documents to use for RAG
NUM_RAG_SOURCES = 5
# Maximum number of urls in generic Google web search or Yahoo Finance web search
MAX_NEWS = 10
# Maximum news per ticker symbol
MAX_NEWS_PER_TICKER = 10


# LLMs by crew
BILLION_PARAMETERS = '70'
MODEL_TEMPLATE = 'sambanova/Meta-Llama-3.3-{billion_parameters}B-Instruct'

GENERAL_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)

CONTEXT_ANALYSIS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
DECOMPOSITION_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
GENERIC_RESEARCH_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
INFORMATION_EXTRACTION_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
RAG_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
REPORT_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
SEC_EDGAR_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_NEWS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_STOCKS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
PANDASAI_MODEL = 'Meta-Llama-3.1-70B-Instruct'

# LLM temperature
TEMPERATURE = 0
