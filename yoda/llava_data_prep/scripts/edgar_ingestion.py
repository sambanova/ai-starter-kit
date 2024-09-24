import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from yoda.llava_data_prep.src.edgar_ingestion import SECTools
from yoda.llava_data_prep.src.table_utils import TableTools

CONFIG_PATH = os.path.join(kit_dir, 'llava_data_prep', 'config.yaml')
DATA_DIRECTORY = os.path.join(kit_dir, 'llava_data_prep', 'sec_data')

parser = argparse.ArgumentParser(description="Download EDGAR reports, detect tables, crop tables, and store.")

parser.add_argument("--name", type=str, help="Name of run", default="")
parser.add_argument("--do-reshape", action="store_true", help="Do reshaping of table or not")
parser.add_argument("--size", type=tuple, help="The output shape if doing reshape", default=(336,336))
parser.add_argument("--det-model", type=str, help="model type for table detection", default="yolo",
                    choices=["yolo", "doclaynet"])


def main() -> None:

    args = parser.parse_args()

    OUTPUT_DIR = DATA_DIRECTORY + "_" + args.name

    sec_tool = SECTools(config=CONFIG_PATH)

    tickers = ["AAPL"]
    form_types = ["10-K"]
    after = "2023-01-01"
    before =  "2024-01-01"

    sec_tool.download_filings(tickers=tickers, 
                            form_types=form_types, 
                            after=after, 
                            before=before, 
                            download_folder=OUTPUT_DIR)

    sec_tool.convert_txt_to_pdf(data_directory=OUTPUT_DIR)

    table_tools = TableTools(do_reshape=args.do_reshape, 
                             size=args.size)

    table_tools.convert_pdf_to_images(data_directory=OUTPUT_DIR)
    if args.det_model == "yolo":
        table_tools.crop_tables(data_directory=OUTPUT_DIR)
    elif args.det_model == "doclaynet":
        table_tools.crop_tables_doclaynet(data_directory=OUTPUT_DIR)

if __name__ == "__main__":
    main()
