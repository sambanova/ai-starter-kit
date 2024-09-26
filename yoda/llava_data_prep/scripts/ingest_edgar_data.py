import os
import sys
import argparse
import logging
import yaml


current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from yoda.llava_data_prep.src.edgar_ingestion import SECTools # type: ignore
from yoda.llava_data_prep.src.table_utils import TableTools # type: ignore

CONFIG_PATH = os.path.join(kit_dir, 'llava_data_prep', 'config.yaml')
DATA_DIRECTORY = os.path.join(kit_dir, 'llava_data_prep', 'sec_data')
LOGS_PATH = os.path.join(DATA_DIRECTORY, "logs")

parser = argparse.ArgumentParser(description="Download EDGAR reports, detect tables, crop tables, and store.")

parser.add_argument("--name", type=str, help="Name of run", default="defaults")

def main() -> None:
    
    try:
        with open(CONFIG_PATH, 'r') as file:
            configs =  yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'The YAML configuration file {CONFIG_PATH} was not found.')
    except yaml.YAMLError as e:
        raise RuntimeError(f'Error parsing YAML file: {e}')

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH, exist_ok=True)

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(LOGS_PATH, f"{args.name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f'Ingesting SEC EDGAR data with configs: ' +
                 f'{configs["sec"]}')

    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(DATA_DIRECTORY, args.name)

    logging.info(f"Running ingest_edgar_data.py with args: " +
                 f"{args} to output directory: " +
                 f"{OUTPUT_DIR}")

    sec_tool = SECTools(config_path=CONFIG_PATH)

    sec_tool.download_filings(download_folder=OUTPUT_DIR)
    sec_tool.convert_txt_to_pdf(data_directory=OUTPUT_DIR)

    logging.info("Downloaded filings and converted to pdf.")
    logging.info("Converting pdfs to images.")

    table_tools = TableTools(config_path=CONFIG_PATH)

    table_tools.convert_pdf_to_images(data_directory=OUTPUT_DIR)
    table_tools.crop_tables(data_directory=OUTPUT_DIR)

if __name__ == "__main__":
    main()
