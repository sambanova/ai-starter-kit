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

# Main cache directories
CACHE_DIR = os.path.join(kit_dir, 'streamlit/cache/')
HISTORY_PATH = os.path.join(CACHE_DIR, 'chat_history.txt')
PDF_GENERATION_DIRECTORY = os.path.join(CACHE_DIR, 'pdf_generation/')
STOCK_QUERY_PATH = os.path.join(CACHE_DIR, 'stock_query.txt')
DB_QUERY_PATH = os.path.join(CACHE_DIR, 'db_query.txt')
YFINANCE_NEWS_PATH = os.path.join(CACHE_DIR, 'yfinance_news.txt')
FILINGS_PATH = os.path.join(CACHE_DIR, 'filings.txt')
PDF_RAG_PATH = os.path.join(CACHE_DIR, 'pdf_rag.txt')

# Main source directories
SOURCE_DIR = os.path.join(CACHE_DIR, 'sources/')
DB_PATH = os.path.join(SOURCE_DIR, 'stock_database.db')

# Main figures directories
STOCK_QUERY_FIGURES_DIR = os.path.join(CACHE_DIR, 'stock_query_figures/')
HISTORY_FIGURES_DIR = os.path.join(CACHE_DIR, 'history_figures/')
DB_QUERY_FIGURES_DIR = os.path.join(CACHE_DIR, 'db_query_figures/')

# Main text processing constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 256
CHUNK_OVERLAP = 64

# Main colors
SAMBANOVA_ORANGE = (238, 118, 36)
