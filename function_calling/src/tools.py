import operator
import os
import re
import sys
from datetime import datetime
from typing import Optional, Union

from dotenv import load_dotenv
from langchain_community.llms.sambanova import SambaStudio
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


load_dotenv(os.path.join(repo_dir, '.env'))

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
    return f'The following errors occurred during Calculator tool execution: `{error.args}`'


# tool definition
calculator = StructuredTool.from_function(
    func=calculator,
    args_schema=CalculatorSchema,
    handle_tool_error=_handle_error,  # set as True if you want the tool to trow a generic ToolError message "Tool execution error"
)

## Python standard shell, or REPL (Read-Eval-Print Loop)


# tool schema
class ReplSchema(BaseModel):
    "A Python shell. Use this to execute python commands. Input should be a valid python commands and expressions. If you want to see the output of a value, you should print it out with `print(...)`, if you need a specific module you should import it."

    command: str = Field(..., description='python code to evaluate')


# tool definition
python_repl = PythonREPL()
python_repl = Tool(
    name='python_repl',
    description='A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.',
    func=python_repl.run,
    args_schema=ReplSchema,
)


## SQL tool
# tool schema
class QueryDBSchema(BaseModel):
    "A database querying tool. Use this to generate sql querys and retrieve the results from a database. Input should be a natural language question to the db."

    query: str = Field(..., description='python code to execute to evaluate')


def sql_finder(text):
    sql_code_pattern = re.compile(r'```sql\s+(.*?)\s+```', re.DOTALL)
    match = sql_code_pattern.search(text)
    if match:
        query = match.group(1)
        return query
    else:
        raise Exception('No sql code found in LLM generation')


@tool(args_schema=QueryDBSchema)
def query_db(query):
    """A database querying tool. Use this to generate sql querys and retrieve the results from a database. Input should be a natural language question to the db."""

    # llm = Sambaverse(
    #     sambaverse_model_name='Meta/Meta-Llama-3-8B-Instruct',
    #     streaming=True,
    #     model_kwargs={
    #         'max_tokens_to_generate': 512,
    #         'select_expert': 'Meta-Llama-3-8B-Instruct',
    #         'temperature': 0.0,
    #         'repetition_penalty': 1.0,
    #         'top_k': 1,
    #         'top_p': 1.0,
    #         'do_sample': False,
    #     },
    # )
    llm = SambaStudio(
        streaming=True,
        model_kwargs={
            'max_tokens_to_generate': 512,
            'select_expert': 'Meta-Llama-3-8B-Instruct',
            'temperature': 0.0,
            'repetition_penalty': 1.0,
            'top_k': 1,
            'top_p': 1.0,
            'do_sample': False,
        },
    )

    db_path = os.path.join(kit_dir, 'data/chinook.db')
    db_uri = f'sqlite:///{db_path}'
    db = SQLDatabase.from_uri(db_uri)

    prompt = PromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        
        {table_info}
        
        Generate a query Using valid SQLite to answer the following questions for the tables provided above.
        <|eot_id|><|start_header_id|>user<|end_header_id|>\
            
        {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )

    query_generation_chain = prompt | llm | RunnableLambda(sql_finder)
    table_info = db.get_table_info()
    query = query_generation_chain.invoke({'input': query, 'table_info': table_info})
    print(query)
    query_executor = QuerySQLDataBaseTool(db=db)
    result = query_executor.invoke(query)

    return result
