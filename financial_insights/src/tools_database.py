import datetime
import json
import logging
import re
from typing import Any, Dict, List, Union

import pandas
import streamlit
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

# from langchain_core.pydantic_v1 import BaseModel, Field
from llama_index.core.bridge.pydantic import BaseModel, Field
from pandasai import SmartDataframe
from pandasai.connectors import SqliteConnector
from sqlalchemy import Inspector, create_engine

from financial_insights.src.tools import coerce_str_to_list, convert_data_to_frame, extract_yfinance_data
from financial_insights.streamlit.constants import *
from utils.model_wrappers.api_gateway import APIGateway


class DatabaseSchema(BaseModel):
    """Create a SQL database for a list of stocks/companies."""

    symbol_list: List[str] | str = Field('List of stock ticker symbols for which to create the SQL database.')
    start_date: datetime.date = Field('Start date.')
    end_date: datetime.date = Field('End date.')


@tool(args_schema=DatabaseSchema)
def create_stock_database(
    symbol_list: List[str] | str,
    start_date: datetime.date = datetime.datetime.today().date() - datetime.timedelta(days=365),
    end_date: datetime.date = datetime.datetime.today().date(),
) -> Dict[str, List[str]]:
    """
    Create a SQL database for a list of stocks/companies.

    Args:
        symbol_list: List of stock ticker symbols for which to create the SQL database.
        start_date: Start date for the historical data.
        end_date: End date for the historical data.

    Returns:
        A dictionary with company symbols as keys and a list of SQL table names as values.

    Raises:
        ValueError: If `start_date` is greater than or equal to `end_date`.
    """
    # Check inputs
    assert isinstance(
        symbol_list, (list, str)
    ), f'`symbol_list` must be a list or a string. Got type{(type(symbol_list))}.'

    # If `symbol_list` is a string, coerce it to a list of strings
    symbol_list = coerce_str_to_list(symbol_list)

    assert all([isinstance(name, str) for name in symbol_list]), TypeError(
        '`company_names_list` must be a list of strings.'
    )

    if start_date > end_date or (end_date - datetime.timedelta(days=365)) < start_date:
        raise ValueError('Start date must be before the end date.')

    # Extract yfinance data
    company_data_dict = dict()
    for symbol in symbol_list:
        company_data_dict[symbol] = extract_yfinance_data(symbol, start_date, end_date)

    # Create SQL database
    company_tables = store_company_dataframes_to_sqlite(db_name=DB_PATH, company_data_dict=company_data_dict)

    return company_tables


def store_company_dataframes_to_sqlite(
    db_name: str, company_data_dict: Dict[str, Union[pandas.DataFrame, Dict[Any, Any]]]
) -> Dict[str, list[str]]:
    """
    Store multiple dataframes for each company into an SQLite database.

    Args:
        db_name: The name of the SQLite database file.
        company_data_dict: Dictionary where the key is the company name,
            and the value is another dictionary or a `pandas.DataFrame` containing
            dataframes with their corresponding purpose/type.

    Returns:
        A dictionary with company symbols as keys and a list of SQL table names as values.
    """
    # Connect to the SQLite database
    engine = create_engine(f'sqlite:///{DB_PATH}')

    # Create a dictionary with company names as keys and SQL tables as values
    company_tables: Dict[str, list[str]] = dict()

    # Process each company
    for company, company_data in company_data_dict.items():
        # Ensure that the company name is SQLite-friendly
        company_base_name = company.replace(' ', '_').lower()

        # Initialize a list of SQL table names
        company_tables[company] = list()

        for df_name, data in company_data.items():
            # Build a table name using the company symbol and the dataframe purpose/type
            table_name = f'{company_base_name}_{df_name}'
            try:
                df = convert_data_to_frame(data, df_name)
            except:
                logging.warning(f'Could not convert {df_name} to `pandas.DataFrame`.')
                continue

            # Make sure the column names are SQLite-friendly
            for column in df.columns:
                df = df.rename({column: f'{column}'.replace(' ', '_')}, axis='columns')

            # Convert list-type and dict-type entries to JSON strings
            df = df.applymap(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

            # Store the dataframe in an SQLite database table
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logging.info(f"DataFrame '{df_name}' for {company} stored in table '{table_name}'.")

            # Populated company tables list with table name
            company_tables[company].append(table_name)

    return company_tables


class QueryDatabaseSchema(BaseModel):
    """Query a SQL database for a list of stocks/companies."""

    user_query: str = Field('Query to be performed on the database')
    symbol_list: List[str] | str = Field('List of stock ticker symbols.')
    method: str = Field('Method to be used in query. Either "text-to-SQL" or "PandasAI-SqliteConnector"')


@tool(args_schema=QueryDatabaseSchema)
def query_stock_database(
    user_query: str, symbol_list: List[str] | str, method: str
) -> Union[Any, Dict[str, str | List[str]]]:
    """
    Query a SQL database for a list of stocks/companies.

    Args:
        user_query: Query to be performed on the database.
        symbol_list: List of stock ticker symbols.
        method: Method to be used in query. Either "text-to-SQL" or "PandasAI-SqliteConnector".

    Returns:
        The result of the query.

    Raises:
        TypeError: If `user_query`, `symbol_list` and `method` are not strings.
        ValueError: If `method` is not one of `text-to-SQL` or `PandasAI-SqliteConnector`.
        Exception: If `symbol_list` is an empty string.
    """
    # Checks the inputs
    assert isinstance(user_query, str), TypeError(f'`symbol_list` must be of type str. Got {(type(user_query))}')

    assert isinstance(symbol_list, (list, str)), TypeError(
        f'`symbol_list` must be a list or a string. Got type {type(symbol_list)}.'
    )

    # If `symbol_list` is a string, coerce it to a list of strings
    symbol_list = coerce_str_to_list(symbol_list)

    assert all([isinstance(name, str) for name in symbol_list]), TypeError('`symbol_list` must be a list of strings.')

    assert isinstance(method, str), TypeError(f'method must be of type str. Got {type(method)}')
    assert method in [
        'text-to-SQL',
        'PandasAI-SqliteConnector',
    ], ValueError(f'Invalid method {method}')

    if method == 'text-to-SQL':
        return query_stock_database_sql(user_query, symbol_list)
    elif method == 'PandasAI-SqliteConnector':
        return query_stock_database_pandasai(user_query, symbol_list)
    else:
        raise ValueError(f'`method` should be either `text-to-SQL` or `PandasAI-SqliteConnector`. Got {method}')


def query_stock_database_sql(user_query: str, symbol_list: List[str]) -> Dict[str, str | List[str]]:
    """
    Query a SQL database for a list of stocks/companies.

    Args:
        user_query: Query to be performed on the database.
        symbol_list: List of stock ticker symbols.

    Returns:
        The result of the query.
    """
    # Prompt template for the SQL queries
    prompt = PromptTemplate.from_template(
        """<|begin_of_text|><|start_of_system|> 
        
        {selected_schemas}
        
        Generate a valid SQLite query to answer the following question,
        based on the summarized table schemas provided above.
        Do not assume any values in the database tables before generating the SQL query. 
        Always generate SQL code that queries exactly what is asked. 
        The queries must be formatted as follows: 
        
        ```sql
        query
        ```
        
        Example:
        
        ```sql
        SELECT * FROM mainTable;
        ```
        
        <|end_of_system|><|start_of_user|>
            
        {input}
        <|end_of_user|><|start_of_assistant|>"""
    )

    # Chain that receives the natural language input and the table schemas, invoke the LLM,
    # and finally execute the SQL finder method, retrieving only the filtered SQL query
    query_generation_chain = prompt | streamlit.session_state.fc.llm | RunnableLambda(sql_finder)

    # Extract the names of the SQL tables that are relevant to the user query
    selected_tables = select_database_tables(user_query, symbol_list)
    selected_schemas = get_table_summaries_from_names(selected_tables)

    # Generate the SQL query
    query: str = query_generation_chain.invoke({'selected_schemas': selected_schemas, 'input': user_query})

    # Split the SQL query into multiple queries
    queries = query.split(';')
    queries = [query for query in queries if len(query) > 0]

    # Create a SQL database engine and connect to it using the selected tables
    engine = create_engine(f'sqlite:///{DB_PATH}')
    db = SQLDatabase(engine=engine, include_tables=selected_tables)

    # Instantiate the SQL executor
    query_executor = QuerySQLDataBaseTool(db=db)
    logging.warning(
        'Executing model-generated SQL queries. There are inherent risks in doing this. '
        + 'Make sure that your database connection permissions are always scoped '
        + 'as narrowly as possible for your chain. '
        + 'This will mitigate though not eliminate the risks of building a model-driven system. '
        + 'For more on general security best practices, see [here]{https://python.langchain.com/v0.1/docs/security/}.'
    )

    # Invoke the SQL executor on each query
    results = []
    for query in queries:
        if len(query) > 0:
            results.append(query_executor.invoke(query))

    message = '\n'.join(
        [f'Query:\n{query}\nexecuted with result:\n{result}' for query, result in zip(queries, results)]
    )

    response_dict: Dict[str, str | List[str]] = dict()
    response_dict['queries'] = queries
    response_dict['results'] = results
    response_dict['message'] = message
    return response_dict


def sql_finder(text: str) -> Any:
    """
    This function will search for a SQL query or code in the text and return the SQL query/code as a string.

    Args:
        text: the text to search for SQL queries/codes.

    Returns:
        The SQL query/code as a string.

    Raises:
        Exception: If no SQL query was found in the input text.
    """

    # regex for finding sql_code_pattern with format:
    # ```sql
    #    <query>
    # ```
    sql_code_pattern = re.compile(r'```sql\s+(.*?)\s+```', re.DOTALL)
    match = sql_code_pattern.search(text)
    if match is not None:
        query = match.group(1)
        return query
    else:
        # regex for finding sql_code_pattern with format:
        # ```
        # <quey>
        # ```
        code_pattern = re.compile(r'```\s+(.*?)\s+```', re.DOTALL)
        match = code_pattern.search(text)
        if match is not None:
            query = match.group(1)
            return query
        else:
            raise Exception(f'No SQL code found in LLM generation from {text}.')


def query_stock_database_pandasai(user_query: str, symbol_list: List[str]) -> Any:
    """
    Query a SQL database for a list of stocks/companies using 'pandasai.connectors.SqliteConnector'.

    Args:
        user_query: The user query.
        symbol_list: The list of stocks/companies.

    Returns:
        The answer to the user query.
    """
    response: Dict[str, List[str]] = dict()
    for symbol in symbol_list:
        # Extract the SQL tables that are relevant to the user query.
        selected_tables = select_database_tables(user_query, [symbol])

        # Add instructions on how to deal with the plots
        user_query += (
            '\nPlease add information on which table is being used (table name) as a text string or in the plot title.'
        )

        response[symbol] = list()
        for table in selected_tables:
            # Instantiate the connector to the SQL database
            connector = SqliteConnector(
                config={
                    'database': DB_PATH,
                    'table': table,
                }
            )

            # Instantiate the `pandasai.SmartDataframe` dataframe
            df = SmartDataframe(
                connector,
                config={
                    'llm': streamlit.session_state.fc.llm,
                    'open_charts': False,
                    'save_charts': True,
                    'save_charts_path': DB_QUERY_FIGURES_DIR,
                },
            )
            # Append the response for the given company symbol
            response[symbol].append(df.chat(user_query))

    return response


class TableNames(BaseModel):
    """Output object for the relevant table names."""

    table_names: List[str] = Field(description='List of the most relevant table names for the user query.')


def select_database_tables(user_query: str, symbol_list: List[str]) -> List[str]:
    """
    Selects the SQL tables that are relevant for the user query.

    Args:
        user_query: The user query.
        symbol_list: List of company symbols.

    Returns:
        List of SQL tables that are relevant for the user query.
    """
    # Get a text symmary of the SQL tables that are relevant for each company symbol
    summary_text = get_table_summaries_from_symbols(symbol_list)

    # The output parser
    parser = PydanticOutputParser(pydantic_object=TableNames)

    # The prompt template
    prompt_template = (
        'Consider the following table summaries with table names and table columns:\n{summary_text}\n'
        'Which are the most relevant tables to the following query?\n'
        '{user_request}\n'
        '{format_instructions}'
    )

    # The prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['user_request', 'summary_text'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    # Invoke the chain with the user query and the table summaries
    max_tokens_to_generate_list = list({streamlit.session_state.fc.llm_info['max_tokens_to_generate'], 1024, 256, 128})
    # Bound the number of tokens to generate based on the config value
    max_tokens_to_generate_list = [
        elem
        for elem in max_tokens_to_generate_list
        if elem < streamlit.session_state.fc.llm_info['max_tokens_to_generate']
    ]

    # Call the LLM for each number of tokens to generate
    # Return the first valid response
    for item in max_tokens_to_generate_list:
        try:
            # Instantiate the LLM
            llm = APIGateway.load_llm(
                type=streamlit.session_state.fc.llm_info['api'],
                streaming=False,
                coe=streamlit.session_state.fc.llm_info['coe'],
                do_sample=streamlit.session_state.fc.llm_info['do_sample'],
                max_tokens_to_generate=item,
                temperature=streamlit.session_state.fc.llm_info['temperature'],
                select_expert=streamlit.session_state.fc.llm_info['select_expert'],
                process_prompt=False,
                sambaverse_model_name=streamlit.session_state.fc.llm_info['sambaverse_model_name'],
            )

            # The chain
            chain = prompt | llm | parser
            response = chain.invoke({'user_request': user_query, 'summary_text': summary_text})
            assert isinstance(response.table_names, list) and all(
                [isinstance(elem, str) for elem in response.table_names]
            ), 'Invalid response'
            return response.table_names
        except:
            pass

    return list()


def get_table_summaries_from_symbols(symbol_list: List[str]) -> str:
    """
    Get a text symmary of the SQL tables that are relevant for each company symbol.

    The summary is a JSON string that maps SQL table names to their column names.

    Args:
        symbol_list: The list of company symbols.

    Returns:
        The text summary of all the SQL table, by company symbol.

    Raises:
        Exception: If there is no SQL table in the database.
    """
    # Instantiate the inspector for the database
    inspector = Inspector.from_engine(create_engine('sqlite:///' + DB_PATH))

    # Get the list of SQL tables in the database
    tables_names = inspector.get_table_names()

    # Check that there are SQL tables
    assert len(tables_names) > 0, 'No SQL tables found.'

    table_summaries = {}
    for table in tables_names:
        # Extract the first word from the name string to get the symbol
        table_symbol = table.split('_')[0]

        # Check if the symbol is in the list of symbols to be queried
        if table_symbol not in [symbol.lower() for symbol in symbol_list]:
            continue

        # Get the columns of each SQL table
        columns = inspector.get_columns(table)
        # Get the column names
        column_names = [col['name'] for col in columns]

        # Summarize the content of the table based on its column names
        table_summaries[table] = ', '.join(column_names)

    # Compose the text summary in a JSON string
    summary_text = json.dumps(table_summaries)
    return summary_text


def get_table_summaries_from_names(table_names: List[str]) -> str:
    """
    Get a text summary of SQL tables by their names.

    The summary is a JSON string that maps SQL table names to their column names.

    Args:
        table_names: List of SQL table names to be summarized.

    Returns:
        A text summary of SQL tables by their names.
    """
    inspector = Inspector.from_engine(create_engine('sqlite:///' + DB_PATH))
    inspected_tables_names = inspector.get_table_names()
    inspected_tables_names_symbols = [
        inspected_table.split('_')[0].lower() for inspected_table in inspected_tables_names
    ]

    table_summaries = {}
    for table in table_names:
        # Extract the first word from the name string to get the symbol
        table_symbol = table.split('_')[0].lower()

        # Check if the symbol is in the list of symbols to be queried
        if table_symbol not in [symbol.lower() for symbol in inspected_tables_names_symbols]:
            continue

        columns = inspector.get_columns(table)
        column_names = [col['name'] for col in columns]

        # Summarize the content of the table based on its column names
        table_summaries[table] = ', '.join(column_names)

    summary_text = json.dumps(table_summaries)
    return summary_text
