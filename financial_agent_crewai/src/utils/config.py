from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# Maximum number of words per section
MAX_SECTION_WORDS = 1000

# User query
USER_QUERY = 'What were Google expenditure records in 2023?'
# USER_QUERY = 'What were the main differences in expenditure records between Apple and Google in 2023?'

# Comparison query
COMPARISON_QUERY = (
    'Please write a conlcusion for the provided context, '
    f'specifically focusing around the user query: {USER_QUERY}.\n'
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
sambanova_model_template = 'sambanova/Meta-Llama-3.1-{billion_parameters}B-Instruct'
GENERAL_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)

CONTEXT_ANALYSIS_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
DECOMPOSITION_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
GENERIC_RESEARCH_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
INFORMATION_EXTRACTION_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
RAG_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
QA_MODEL = f'Meta-Llama-3.1-{BILLION_PARAMETERS}B-Instruct'
REPORT_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
SEC_EDGAR_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_NEWS_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)
YFINANCE_STOCKS_MODEL = sambanova_model_template.format(billion_parameters=BILLION_PARAMETERS)

# LLM temperature
TEMPERATURE = 0
