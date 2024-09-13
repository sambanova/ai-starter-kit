import os
import sys

# Main directories
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

# Main config file
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

# Cache directory template
CACHE_DIR = os.path.join(kit_dir, 'streamlit/cache/')

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

# LLM constants
MAX_RETRIES = 3

# STOCK INFO
YFINANCE_COLUMNS_JSON = os.path.join(kit_dir, 'streamlit/yfinance_columns.json')

# Define default values for text inputs
DEFAULT_COMPANY_NAME = 'Meta'
DEFAULT_STOCK_QUERY = 'What is the research and development spending trend for Meta?'
DEFAULT_HISTORICAL_STOCK_PRICE_QUERY = 'Meta close value'
DEFAULT_RAG_QUERY = (
    'Have there been changes in strategy, products, and research for Meta? Can you provide some examples?'
)
DEFAULT_PDF_RAG_QUERY = "What conclusions can we draw about Meta's strategy?"
