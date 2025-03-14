import os
from pathlib import Path

# Level of verbosity
VERBOSE = False

# Specify the directory and file path
if Path(os.getcwd()).name == 'ai-starter-kit':
    CACHE_DIR = Path('financial_agent_crewai') / 'cache/'
else:
    CACHE_DIR = Path('cache')

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# Data sources
SOURCE_GENERIC_SEARCH = True
SOURCE_SEC_FILINGS = True
SOURCE_YFINANCE_NEWS = True
SOURCE_YFINANCE_STOCK = True

# User query
USER_QUERY = 'What was the research and development spending trend for Google in 2024?'

# Comparison query
COMPARISON_QUERY = (
    'Please compare the provided contexts, focusing specifically on the user query: '
    f'{USER_QUERY}.\n'
    f'Do not exceed {MAX_SECTION_WORDS/2} words.'
)
# Number of documents to use for RAG
NUM_RAG_SOURCES = 10
# Maximum number of urls in generic Google web search or Yahoo Finance web search
MAX_NEWS = 10
# Maximum news per ticker symbol
MAX_NEWS_PER_TICKER = 10


# LLMs by crew
BILLION_PARAMETERS = '70'
MODEL_TEMPLATE = 'sambanova/Meta-Llama-3.3-{billion_parameters}B-Instruct'

CONTEXT_ANALYSIS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
DECOMPOSITION_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
GENERIC_RESEARCH_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
INFORMATION_EXTRACTION_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
RAG_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
REPORT_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
SEC_EDGAR_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_NEWS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_STOCKS_MODEL = MODEL_TEMPLATE.format(billion_parameters=BILLION_PARAMETERS)
PANDASAI_MODEL = 'Meta-Llama-3.3-70B-Instruct'

# LLM temperature
TEMPERATURE = 0
