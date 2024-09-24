import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from yoda.llava_data_prep.src.edgar_ingestion import SECTools
from yoda.llava_data_prep.src.table_utils import TableTools

CONFIG_PATH = os.path.join(kit_dir, 'llava_data_prep', 'config.yaml')
DATA_DIRECTORY = os.path.join(kit_dir, 'llava_data_prep', 'sec_data_yolo')

sec_tool = SECTools(config=CONFIG_PATH)

tickers = ["AAPL"]
form_types = ["10-K"]
after = "2023-01-01"
before =  "2024-01-01"

sec_tool.download_filings(tickers=tickers, 
                          form_types=form_types, 
                          after=after, 
                          before=before, 
                          download_folder=DATA_DIRECTORY)

sec_tool.convert_txt_to_pdf(data_directory=DATA_DIRECTORY)

table_tools = TableTools()

table_tools.convert_pdf_to_images(data_directory=DATA_DIRECTORY)
table_tools.crop_tables(data_directory=DATA_DIRECTORY)