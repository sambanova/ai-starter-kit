from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# User query
# USER_QUERY = 'What is the product strategy of Google as per its 10-K filing in 2024?'
USER_QUERY = 'What were the main differences in spending trends between Apple and Google in 2023?'

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
sambanova_model_template = 'sambanova/Meta-Llama-3.1-{billion_parameters}B-Instruct'
GENERAL_MODEL = 'sambanova/Meta-Llama-3.1-70B-Instruct'

CONTEXT_ANALYSIS_MODEL = sambanova_model_template.format(billion_parameters='70')
DECOMPOSITION_MODEL = sambanova_model_template.format(billion_parameters='70')
GENERIC_RESEARCH_MODEL = sambanova_model_template.format(billion_parameters='70')
INFORMATION_EXTRACTION_MODEL = sambanova_model_template.format(billion_parameters='70')
RAG_MODEL = sambanova_model_template.format(billion_parameters='70')
QA_MODEL = 'Meta-Llama-3.1-70B-Instruct'
REPORT_MODEL = sambanova_model_template.format(billion_parameters='70')
SEC_EDGAR_MODEL = sambanova_model_template.format(billion_parameters='70')
YFINANCE_NEWS_MODEL = sambanova_model_template.format(billion_parameters='70')
YFINANCE_STOCKS_MODEL = sambanova_model_template.format(billion_parameters='70')

# LLM temperature
TEMPERATURE = 0
