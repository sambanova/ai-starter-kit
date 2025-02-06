from pathlib import Path

# Level of verbosity
VERBOSE = False

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')
YFINANCE_STOCKS_DIR = CACHE_DIR / 'yfinance_stocks'
PANDASAI_CAHE_DIR = Path('cache')

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# User query
USER_QUERY = 'Can you give me a breakdown analysis of Google expenditures for the last quarter of 2024?'
# USER_QUERY = 'What were the main differences in expenditure records between Apple and Google in 2023?'
# USER_QUERY = 'What are the latest close value for Google for the last 3 months?'

# Comparison query
COMPARISON_QUERY = (
    'Please compare the provided contexts, focusing specifically on the user query: '
    f'{USER_QUERY}.\n'
    f'Do not exceed {MAX_SECTION_WORDS/2} words.'
)
# Number of documents to use for RAG
NUM_RAG_SOURCES = 5
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
BILLION_PARAMETERS = '70'
PROVIDER = 'sambanova'
MODEL_TEMPLATE = '{provider}/Meta-Llama-3.1-{billion_parameters}B-Instruct'
MODEL_NAME_TEMPLATE = 'Meta-Llama-3.1-{billion_parameters}B-Instruct'

GENERAL_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)

CONTEXT_ANALYSIS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
DECOMPOSITION_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
GENERIC_RESEARCH_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
INFORMATION_EXTRACTION_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
RAG_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
REPORT_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
SEC_EDGAR_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
YFINANCE_NEWS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
YFINANCE_STOCKS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
PANDASAI_MODEL = MODEL_NAME_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)

# LLM temperature
TEMPERATURE = 0
