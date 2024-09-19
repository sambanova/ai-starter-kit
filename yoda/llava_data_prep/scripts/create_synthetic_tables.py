import os
import json
import sys
import argparse
import pprint
import logging
from dotenv import load_dotenv

load_dotenv()

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

pp = pprint.PrettyPrinter(width=80)

CONFIG_PATH = os.path.join(kit_dir, "llava_data_prep", "config.yaml")
PREP_PATH = os.path.join(kit_dir, "llava_data_prep")
LOGS_PATH = os.path.join(kit_dir, "llava_data_prep", "logs")

from yoda.llava_data_prep.src.table_utils import TableAugmentor

parser = argparse.ArgumentParser(description="Test and log synthetic table creation.")

parser.add_argument("--name", type=str, help="Name of run", default="llava_prep")
parser.add_argument("--num-its", type=int, help="Number of iterations.", default=25)
parser.add_argument("--split", type=str, help="Split", default="train", choices=["train", "val"]) # TODO: At test and logic

def main() -> None:

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH, exist_ok=True)
    
    # TODO: add split flag and have both a train and validation json, or split based on split arg.  
    with open(os.path.join(kit_dir, "llava_data_prep", "table_templates/table_templates.json"), "r") as f:
        data = json.load(f)
     
    synth_tables: list = []

    for company in data.keys():
        for example in data[company].keys():
            synth_tables.append(data[company][example])

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(LOGS_PATH, f"{args.name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    DATA_DIRECTORY = os.path.join(PREP_PATH, args.name)        

    table_augmentor = TableAugmentor(config_path=CONFIG_PATH)

    try:
        table_augmentor.create_training_data(num_samples=args.num_its,
                                    synth_tables=synth_tables,
                                    split=args.split, 
                                    data_directory=DATA_DIRECTORY)

    except Exception as e:
        logger.error(f"Error encountered during script run: {e}")

if __name__ == "__main__":
    main()
        