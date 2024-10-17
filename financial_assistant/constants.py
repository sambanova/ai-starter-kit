import datetime
import os
import sys
from uuid import uuid4

import yaml

from financial_assistant.src.llm import SambaNovaLLM

# Main directories
kit_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

SESSION_ID = str(uuid4())

# Main config file
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

# Read config file
with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Get the llm information
prod_mode = config['prod_mode']

# Initialize SEC EDGAR credentials
if prod_mode or os.getenv('SEC_API_ORGANIZATION') is None or os.getenv('SEC_API_EMAIL') is None:
    os.environ['SEC_API_ORGANIZATION'] = 'SambaNova'
    os.environ['SEC_API_EMAIL'] = f'user_{SESSION_ID}@sambanova_cloud.com'

# Minutes for scheduled cache deletion
EXIT_TIME_DELTA = 30

# Main text processing, RAG, and web scraping constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
RETRIEVE_HEADLINES = False
TOP_K = 10
MAX_URLS = 1000

# SambaNova
SAMBANOVA_LOGO = 'https://sambanova.ai/hubfs/logotype_sambanova_orange.png'
SAMBANOVA_ORANGE = (238, 118, 36)

# STOCK INFO
YFINANCE_COLUMNS_JSON = os.path.join(kit_dir, 'streamlit/yfinance_columns.json')

# Define default values for text inputs
DEFAULT_COMPANY_NAME = 'Meta'
DEFAULT_DATAFRAME_NAME = 'income_stmt'
DEFAULT_STOCK_QUERY = 'What is the research and development spending trend for Meta?'
DEFAULT_HISTORICAL_STOCK_PRICE_QUERY = 'Meta close value'
DEFAULT_RAG_QUERY = (
    'Have there been changes in strategy, products, and research for Meta? Can you provide some examples?'
)
DEFAULT_PDF_RAG_QUERY = "What conclusions can we draw about Meta's strategy?"
DEFAULT_START_DATE = datetime.datetime.today().date() - datetime.timedelta(days=365)
DEFAULT_END_DATE = datetime.datetime.today().date()
DEFAULT_FILING_TYPE = '10-K'
DEFAULT_FILING_QUARTER = 0
DEFAULT_FILING_YEAR = datetime.datetime.today().date().year - 1
DEFAULT_PDF_TITLE = 'Financial Report'


# Cache directory
CACHE_DIR = os.path.join(kit_dir, 'cache')
if prod_mode:
    CACHE_DIR = os.path.join(CACHE_DIR[:-1] + '_prod_mode', f'cache_{SESSION_ID}')


# Main cache directories
HISTORY_PATH = os.path.join(CACHE_DIR, 'chat_history.txt')
PDF_GENERATION_DIRECTORY = os.path.join(CACHE_DIR, 'pdf_generation')
STOCK_QUERY_PATH = os.path.join(CACHE_DIR, 'stock_query.txt')
DB_QUERY_PATH = os.path.join(CACHE_DIR, 'db_query.txt')
YFINANCE_NEWS_PATH = os.path.join(CACHE_DIR, 'yfinance_news.txt')
FILINGS_PATH = os.path.join(CACHE_DIR, 'filings.txt')
PDF_RAG_PATH = os.path.join(CACHE_DIR, 'pdf_rag.txt')
WEB_SCRAPING_PATH = os.path.join(CACHE_DIR, 'web_scraping.csv')

# Main source directories
SOURCE_DIR = os.path.join(CACHE_DIR, 'sources')
DB_PATH = os.path.join(SOURCE_DIR, 'stock_database.db')
YFINANCE_NEWS_TXT_PATH = os.path.join(SOURCE_DIR, 'yfinance_news_documents.txt')
YFINANCE_NEWS_CSV_PATH = os.path.join(SOURCE_DIR, 'yfinance_news_documents.csv')
PDF_SOURCES_DIR = os.path.join(SOURCE_DIR, 'pdf_sources')

# Main figures directories
STOCK_QUERY_FIGURES_DIR = os.path.join(CACHE_DIR, 'stock_query_figures')
HISTORY_FIGURES_DIR = os.path.join(CACHE_DIR, 'history_figures')
DB_QUERY_FIGURES_DIR = os.path.join(CACHE_DIR, 'db_query_figures')

# `pandasai` cache
PANDASAI_CACHE = os.path.join(os.getcwd(), 'cache')

# Unit tests
TEST_DIR = os.path.join(kit_dir, 'tests/')
TEST_CACHE_DIR = os.path.join(TEST_DIR, 'cache/')

# Instantiate the LLM
sambanova_llm = SambaNovaLLM(config_path=CONFIG_PATH)
