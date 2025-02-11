from pathlib import Path

# Level of verbosity
VERBOSE = False

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')
YFINANCE_STOCKS_DIR = CACHE_DIR / 'yfinance_stocks'
PANDASAI_CAHE_DIR = Path('cache')

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# Data sources
SOURCE_GENERIC_SEARCH = False
SOURCE_SEC_FILINGS = False
SOURCE_YFINANCE_NEWS = False
SOURCE_YFINANCE_STOCK = True

# # User query
USER_QUERY = 'Please give me a breakdown analysis of Google expenditures for the last 6 months?'
# USER_QUERY = 'What were the main differences in expenditure records between Apple and Google in 2023?'
# USER_QUERY = 'What are the latest close value for Google for the last 3 months?'

# # User query to test the Generic Google Search for a single company
# USER_QUERY = (
#     'What are the top trending news stories about Google, and how might these headlines influence investor sentiment?'
# )

# # User query to test the SEC EDGAR Filings Search for a single company
# USER_QUERY = (
#     'From Google’s most recent 10-K filing on the SEC’s EDGAR database, what key risk factors stand out, '
#     'and do they indicate any shifts in the company’s strategic direction'
# )

# # User query to test the Yahoo Finance News Search for a single company
# USER_QUERY = 'How have analysts’ views on Google evolved recently, and what primary catalysts drove those changes?'

# # User query to test the Yahoo Finance Stock Analysis for a single company
# USER_QUERY = 'Can you highlight recent trends in revenue growth and margin changes over the last four quarters?'

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
