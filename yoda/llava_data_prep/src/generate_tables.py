# Helper utility script based on Fenglu Hong @SambaNova Systems' work for table OCR fine tuning.
# Use to quickly iterate on table formatting when building templates.

import os
import sys

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from yoda.llava_data_prep.src.table_utils import TableTools # type: ignore
DATA_DIRECTORY = os.path.join(kit_dir,"synthetic_data")

table_tools = TableTools()

synth_tables: dict = {}

columns: str = "\\begin{tabular}{| l c c r c c r |}\n\\hline\n"

synth_tables["Products_and_Services_Performance"] = """\\textbf{Products and Services Performance}\t\t\t\t\t\t
\t\t\\textbf{Three Months Ended}\t\t\\textbf{Six Months Ended}\t\t
\\hline\t\t\t\t\t\t
****\t\\textbf{April 1,}\t\\textbf{March 26,}\t\t\\textbf{April 1,}\t\\textbf{March 26,}\t
****\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}
\\hline\t\t\t\t\t\t
Net sales by category\t\t\t\t\t\t
*iPhone***\t$***51,334\t$***50,570\t****2%\t$*117,109\t$*122,198\t***(4)%
*Mac***\t****7,168\t****10,435\t***(31)%\t**14,903\t**21,287\t**(30)%
*iPad***\t****6,678\t****7,646\t***(13)%\t**16,066\t**14,894\t****8%
*Wearables, Home and Accessories\t****8,757\t****8,806\t****(1)%\t**22,239\t**23,507\t**(5)%
*Services***\t****20,907\t****19,821\t******5%\t**41,673\t**39,337\t***6%
\\hline\t\t\t\t\t\t
**Total new sales**\t$***94,836\t$***97,278\t*****(3)%\t$**211,990\t$**221,223\t*****(4)%
"""

img_name = "test_table"
data = table_tools.convert_tsv_to_latex(columns, synth_tables["Products_and_Services_Performance"])
path = os.path.join(DATA_DIRECTORY, "tmp")

table_tools.generate_images(folder_name=path,
                            image_name=img_name,
                             data=data)