# Helper utility script based on Fenglu Hong @SambaNova Systems' work for table OCR fine tuning.
# Use to quickly iterate on table formatting when building templates.

import os
import sys
import json

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from yoda.llava_data_prep.src.table_utils import TableTools # type: ignore
DATA_DIRECTORY = os.path.join(kit_dir,"synthetic_data")

table_tools = TableTools()

### Iterate using simple strings as shown below and add examples to 
### the table_templates.json

# synth_tables: dict = {}

# columns: str = "\\begin{tabular}{| l c c |}\n\\hline\n"

# synth_tables["synth_table"] = """\t*****Three Months\t
# \tDecember 31,\tDecember 25,
# \t2022\t2021
# Americas:*********\t\t
# *Net sales\t$**49,278\t$**51,496
# *Operating income\t$**17,864\t$**19,585
# \t\t
# Europe:*********\t\t
# *Net sales\t$**27,681\t$**29,749
# *Operating income\t$**10,017\t$**11,545
# \t\t
# Greater China:*********\t\t
# *Net sales\t$**23,905\t$**25,783
# *Operating income\t$**10,437\t$**11,183
# \t\t
# Japan:*********\t\t
# *Net sales\t$**6,755\t$**7,107
# *Operating income\t$**3,236\t$**3,349
# \t\t
# Rest of Asia Pacific:*********\t\t
# *Net sales\t$**9,535\t$**9,810
# *Operating income\t$**3,851\t$**3,995
# """

with open(os.path.join(kit_dir, "llava_data_prep", "table_templates/table_templates.json"), "r") as f:
    data = json.load(f)

img_name = "test_table"
data = table_tools.convert_tsv_to_latex(data["aapl"]["example7"]["columns"], 
                                        data["aapl"]["example7"]["tsv_formatted"])

# Use below if iterating directly in the script.
# data = table_tools.convert_tsv_to_latex(columns, 
#                                         synth_tables["synth_table"])
path = os.path.join(DATA_DIRECTORY, "tmp/images")

table_tools.generate_images(folder_name=path,
                            image_name=img_name,
                             data=data)