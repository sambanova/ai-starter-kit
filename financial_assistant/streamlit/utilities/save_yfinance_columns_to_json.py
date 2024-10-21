import json
from collections import OrderedDict

from financial_assistant.constants import *
from financial_assistant.src.tools import convert_data_to_frame, extract_yfinance_data
from financial_assistant.src.utilities import get_logger

logger = get_logger()


def save_yfinance_columns_to_json(symbol: str = 'GOOG') -> None:
    """Save the column names of each `yfinance` data source for a given company ticker symbol to a JSON file."""

    # Extract yfinance data
    company_data_dict = extract_yfinance_data(
        'GOOG',
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )

    # Dictionary to hold dataframe columns
    columns_dict = dict()
    for name, data in company_data_dict.items():
        try:
            # Coerce the retrieved data to a `pandas.DataFrame`
            dataframe = convert_data_to_frame(data, name)

            # Save the column names of the DataFrame for later use in a JSON file
            columns_dict[name] = dataframe.columns.tolist()

        except:
            logger.warning(f'Error retrieving {name} data.')

    # Sort the dictionary by keys
    sorted_columns_dict = OrderedDict(sorted(columns_dict.items()))

    # Save the sorted dictionary of columns to a JSON file
    with open(f'{kit_dir}/streamlit/yfinance_columns.json', 'w') as json_file:
        json.dump(sorted_columns_dict, json_file, indent=2)


if __name__ == '__main__':
    save_yfinance_columns_to_json('GOOG')
