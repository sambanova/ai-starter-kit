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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pandasai import SmartDataframe
from pandasai.connectors import SqliteConnector
from sqlalchemy import Inspector, create_engine

from financial_insights.src.tools import convert_data_to_frame, extract_yfinance_data
from financial_insights.streamlit.constants import *


class DatabaseSchema(BaseModel):
    """Create a SQL database for a list of stocks/companies."""

    symbol_list: List[str] = Field('List of stock ticker symbols.')
    start_date: datetime.date = Field('Start date.')
    end_date: datetime.date = Field('End date.')


@tool(args_schema=DatabaseSchema)
def create_stock_database(
    symbol_list: List[str] = list(),
    start_date: datetime.date = datetime.datetime.today().date() - datetime.timedelta(days=365),
    end_date: datetime.date = datetime.datetime.today().date(),
) -> Dict[str, List[str]]:
    """
    Create a SQL database for a list of stocks/companies.

    Args:
        symbol_list: List of stock ticker symbols.
        start_date: Start date for the historical data.
        end_date: End date for the historical data.
    Returns:
        A dictionary with company symbols as keys and a list of SQL table names as values.
    """
    # Check dates
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
            df = convert_data_to_frame(data, df_name)

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
    symbol_list: List[str] = Field('List of stock ticker symbols.')
    method: str = Field('Method to be used in query. Either "text-to-SQL" or "PandasAI-SqliteConnector"')


@tool(args_schema=QueryDatabaseSchema)
def query_stock_database(
    user_query: str, symbol_list: List[str], method: str
) -> Union[Any, Dict[str, str | List[str]]]:
    """
    Query a SQL database for a list of stocks/companies.

    Args:
        user_query: Query to be performed on the database.
        symbol_list: List of stock ticker symbols.
        method: Method to be used in query. Either "text-to-SQL" or "PandasAI-SqliteConnector".
    Returns:
        xxx
    """
    assert method in [
        'text-to-SQL',
        'PandasAI-SqliteConnector',
    ], f'Invalid method {method}'
    assert isinstance(symbol_list, list), 'Symbol List must be a list of strings.'
    assert len(symbol_list) > 0, 'Symbol List must contain at least one symbol: please specify the company to query.'

    if method == 'text-to-SQL':
        return query_stock_database_sql(user_query, symbol_list)
    elif method == 'PandasAI-SqliteConnector':
        return query_stock_database_pandasai(user_query, symbol_list)
    else:
        raise Exception('Invalid method')


def query_stock_database_sql(user_query: str, symbol_list: List[str]) -> Dict[str, str | List[str]]:
    """
    Query a SQL database for a list of stocks/companies.

    Args:
        user_query: Query to be performed on the database.
        symbol_list: List of stock ticker symbols.
    Returns:
        xxx
    """
    # Prompt template for the SQL queries
    prompt = PromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        
        {selected_schemas}
        
        Generate a query using valid SQLite to answer the following questions 
        for the summarized tables schemas provided above.
        Do not assume the values on the database tables before generating the SQL query, 
        always generate a SQL code that queries what is asked. 
        The query must be in the format: ```sql\nquery\n```
        
        Example:
        
        ```sql
        SELECT * FROM mainTable;
        ```
        
        <|eot_id|><|start_header_id|>user<|end_header_id|>\
            
        {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
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
    logging.warning('')

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
    """Search in a string for a SQL query or code with format"""

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
            raise Exception('No SQL code found in LLM generation')


def query_stock_database_pandasai(user_request: str, symbol_list: List[str]) -> Any:
    """Query a SQL database for a list of stocks/companies."""

    response: Dict[str, List[str]] = dict()
    for symbol in symbol_list:
        selected_tables = select_database_tables(user_request, [symbol])

        user_request += (
            '\nPlease add information on which table is being used (table name) as a text string or in the plot title.'
        )

        response[symbol] = list()
        for table in selected_tables:
            # selecteed_tables = select_database_tables(user_request, symbol_list)
            connector = SqliteConnector(
                config={
                    'database': DB_PATH,
                    'table': table,
                }
            )

            df = SmartDataframe(
                connector,
                config={
                    'llm': streamlit.session_state.fc.llm,
                    'open_charts': False,
                    'save_charts': True,
                    'save_charts_path': DB_QUERY_FIGURES_DIR,
                },
            )
            response[symbol].append(df.chat(user_request))

    return response


class TableNames(BaseModel):
    table_names: List[str] = Field(description='List of the most relevant table names for the user query.')


def select_database_tables(user_request: str, symbol_list: List[str]) -> List[str]:
    summary_text = get_table_summaries_from_symbols(symbol_list)

    parser = PydanticOutputParser(pydantic_object=TableNames)  # type: ignore
    prompt_template = (
        'Consider the following table summaries with table names and table columns:\n{summary_text}\n'
        'Which are the most relevant tables to the following query?\n'
        '{user_request}\n'
        '{format_instructions}'
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['user_request', 'summary_text'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    chain = prompt | streamlit.session_state.fc.llm | parser

    # Get response from llama3
    response = chain.invoke({'user_request': user_request, 'summary_text': summary_text})
    return response.table_names  # type: ignore


def get_table_summaries_from_symbols(symbol_list: List[str]) -> str:
    """Get a list of available SQL tables."""
    inspector = Inspector.from_engine(create_engine('sqlite:///' + DB_PATH))
    tables_names = inspector.get_table_names()

    assert len(tables_names), 'No SQL tables found.'

    table_summaries = {}
    for table in tables_names:
        # Extract the first word from the name string to get the symbol
        table_symbol = table.split('_')[0]

        # Check if the symbol is in the list of symbols to be queried
        if table_symbol not in [symbol.lower() for symbol in symbol_list]:
            continue

        columns = inspector.get_columns(table)
        column_names = [col['name'] for col in columns]

        # Summarize the content of the table based on its column names
        table_summaries[table] = ', '.join(column_names)

    summary_text = json.dumps(table_summaries)
    return summary_text


def get_table_summaries_from_names(table_names: List[str]) -> str:
    """Get a list of available SQL tables."""
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
