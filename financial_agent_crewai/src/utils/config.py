from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# User query
USER_QUERY = 'What were Google expenditures for 2023 according to its SEC Edgar filings?'
# USER_QUERY = 'What were the main differences in expenditure records between Apple and Google in 2023?'

# Comparison query
COMPARISON_QUERY = (
    'Please compare the provided contexts, focusing specifically on the user query: '
    f'{USER_QUERY}.\n'
    f'Do not exceed {MAX_SECTION_WORDS/2} words.'
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
BILLION_PARAMETERS = '70'
PROVIDER = 'sambanova'
MODEL_TEMPLATE = '{provider}/Meta-Llama-3.1-{billion_parameters}B-Instruct'

GENERAL_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)

CONTEXT_ANALYSIS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
DECOMPOSITION_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
GENERIC_RESEARCH_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
INFORMATION_EXTRACTION_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
RAG_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
QA_MODEL = f'Meta-Llama-3.1-{BILLION_PARAMETERS}B-Instruct'
REPORT_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
SEC_EDGAR_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
YFINANCE_NEWS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)
YFINANCE_STOCKS_MODEL = MODEL_TEMPLATE.format(provider=PROVIDER, billion_parameters=BILLION_PARAMETERS)

# LLM temperature
TEMPERATURE = 0
