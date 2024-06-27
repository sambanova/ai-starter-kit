import operator
import os
import re
import sys
from datetime import datetime
from typing import Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, Tool, ToolException, tool
from langchain_experimental.utilities import PythonREPL

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

load_dotenv(os.path.join(repo_dir, '.env'))


## Get configs for tools
def get_config_info(config_path: str) -> tuple[dict]:
    """
    Loads json config file
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    tools_info = config['tools']

    return (tools_info,)


##Get time tool


# tool schema
class GetTimeSchema(BaseModel):
    """Returns current date, current time or both."""

    kind: Optional[str] = Field(description='kind of information to retrieve "date", "time" or "both"')


# definition using @tool decorator
@tool(args_schema=GetTimeSchema)
def get_time(kind: str = 'both') -> str:
    """Returns current date, current time or both.

    Args:
        kind: date, time or both
    """
    if kind == 'date':
        date = datetime.now().strftime('%d/%m/%Y')
        return f'Current date: {date}'
    elif kind == 'time':
        time = datetime.now().strftime('%H:%M:%S')
        return f'Current time: {time}'
    else:
        date = datetime.now().strftime('%d/%m/%Y')
        time = datetime.now().strftime('%H:%M:%S')
        return f'Current date: {date}, Current time: {time}'


## Calculator Tool


class CalculatorSchema(BaseModel):
    """allow calculation of only basic operations: + - * and /
    with a string input expression"""

    expression: str = Field(..., description="expression to calculate, example '12 * 3'")


# function to use in the tool
def calculator(expression: str) -> Union[str, int, float]:
    """
    allow calculation of basic operations
    with a string input expression
    Args:
        expression: expression to calculate
    """
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        'x': operator.mul,
        'X': operator.mul,
        '÷': operator.truediv,
        '/': operator.truediv,
    }
    tokens = re.findall(r'\d+\.?\d*|\+|\-|\*|\/|÷|x|X', expression)

    if len(tokens) == 0:
        raise ToolException(
            f"Invalid expression '{expression}', should only contain one of the following operators + - * x and ÷"
        )

    current_value = float(tokens.pop(0))

    while len(tokens) > 0:
        # The next token should be an operator
        op = tokens.pop(0)

        # The next token should be a number
        if len(tokens) == 0:
            raise ToolException(f"Incomplete expression '{expression}'")
        try:
            next_value = float(tokens.pop(0))

        except ValueError:
            raise ToolException('Invalid number format')

        except:
            raise ToolException('Invalid operation')

        # check division by 0
        if op in ['/', '÷'] and next_value == 0:
            raise ToolException('cannot divide by 0')

        current_value = ops[op](current_value, next_value)

    result = current_value

    return result


# tool error handler
def _handle_error(error: ToolException) -> str:
    """
    tool error handler
    Args:
        error: tool error
    """
    return f'The following errors occurred during Calculator tool execution: `{error.args}`'


# tool definition
calculator = StructuredTool.from_function(
    func=calculator,
    args_schema=CalculatorSchema,
    handle_tool_error=_handle_error,
)  # type: ignore

## Python standard shell, or REPL (Read-Eval-Print Loop)


# tool schema
class ReplSchema(BaseModel):
    (
        'A Python shell. Use this to execute python commands. Input should be a valid python commands and expressions. '
        'If you want to see the output of a value, you should print it out with `print(...)`, '
        'if you need a specific module you should import it.'
    )

    command: str = Field(..., description='python code to evaluate')


# tool definition
python_repl = PythonREPL()
python_repl = Tool(
    name='python_repl',
    description=(
        'A Python shell. Use this to execute python commands. Input should be a valid python command. '
        'If you want to see the output of a value, you should print it out with `print(...)`.'
    ),
    func=python_repl.run,
    args_schema=ReplSchema,
)  # type: ignore


## SQL tool
# tool schema
class QueryDBSchema(BaseModel):
    (
        'A query generation tool. Use this to generate sql queries and retrieve the results from a database. '
        'Do not pass sql queries directly. Input must be a natural language question or instruction.'
    )

    query: str = Field(..., description='natural language question or instruction.')


def sql_finder(text: str) -> str:
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


@tool(args_schema=QueryDBSchema)
def query_db(query: str) -> str:
    """query generation tool. Use this to generate sql queries and retrieve the results from a database.
    Do not pass sql queries directly. Input must be a natural language question or instruction."""

    # get tool configs
    query_db_info = get_config_info(CONFIG_PATH)[0]['query_db']

    # set the llm based in tool configs
    if query_db_info['llm']['api'] == 'sambastudio':
        if query_db_info['llm']['coe']:
            # Using SambaStudio CoE expert as model for generating the SQL Query
            llm = SambaStudio(
                streaming=True,
                model_kwargs={
                    'max_tokens_to_generate': query_db_info['llm']['max_tokens_to_generate'],
                    'select_expert': query_db_info['llm']['select_expert'],
                    'temperature': query_db_info['llm']['temperature'],
                },
            )
        else:
            # Using SambaStudio endpoint as model for generating the SQL Query
            llm = SambaStudio(
                model_kwargs={
                    'max_tokens_to_generate': query_db_info['llm']['max_tokens_to_generate'],
                    'temperature': query_db_info['llm']['temperature'],
                },
            )
    elif query_db_info['llm']['api'] == 'sambaverse':
        # Using Sambaverse expert as model for generating the SQL Query
        llm = Sambaverse(  # type:ignore
            sambaverse_model_name=query_db_info['llm']['sambaverse_model_name'],
            model_kwargs={
                'max_tokens_to_generate': query_db_info['llm']['max_tokens_to_generate'],
                'select_expert': query_db_info['llm']['select_expert'],
                'temperature': query_db_info['llm']['temperature'],
            },
        )
    else:
        raise ValueError(
            f"Invalid LLM API: {query_db_info['llm']['api']}, only 'sambastudio' and'sambaverse' are supported."
        )

    db_path = os.path.join(kit_dir, query_db_info['db']['path'])
    db_uri = f'sqlite:///{db_path}'
    db = SQLDatabase.from_uri(db_uri)

    prompt = PromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        
        {table_info}
        
        Generate a query using valid SQLite to answer the following questions for the summarized tables schemas provided above.
        Do not assume the values on the database tables before generating the SQL query, always generate a SQL that query what is asked. 
        The query must be in the format: ```sql\nquery\n```
        
        Example:
        
        ```sql
        SELECT * FROM mainTable;
        ```
        
        <|eot_id|><|start_header_id|>user<|end_header_id|>\
            
        {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""  # noqa E501
    )

    # Chain that receives the natural language input and the table schema, then pass the teh formatted prompt to the llm
    # and finally execute the sql finder method, retrieving only the filtered SQL query
    query_generation_chain = prompt | llm | RunnableLambda(sql_finder)
    table_info = db.get_table_info()
    query = query_generation_chain.invoke({'input': query, 'table_info': table_info})

    query_executor = QuerySQLDataBaseTool(db=db)
    result = query_executor.invoke(query)

    result = f'Query {query} executed with result {result}'
    return result


## translation tool
# tool schema


# tool schema
class TranslateSchema(BaseModel):
    """Returns translated input sentence to desired language"""

    origin_language: str = Field(description='language of the original sentence')
    final_language: str = Field(description='language to translate the sentence into')
    input_sentence: str = Field(description='sentence to translate')


@tool(args_schema=TranslateSchema)
def translate(origin_language: str, final_language: str, input_sentence: str) -> str:
    """Returns translated input sentence to desired language

    Args:
        origin_language: language of the original sentence
        final_language: language to translate the sentence into
        input_sentence: sentence to translate
    """
    llm = SambaStudio(
        streaming=True,
        model_kwargs={
            'max_tokens_to_generate': 2048,
            'select_expert': 'Meta-Llama-3-8B-Instruct',
            'temperature': 0.2,
        },
    )
    return llm.invoke(f'Translate from {origin_language} to {final_language}: {input_sentence}')
