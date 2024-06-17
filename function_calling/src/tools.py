import os
import re
import sys
import operator
from datetime import datetime
from dotenv import load_dotenv
from typing import  Optional, Union
from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL


##Get time tool

# tool schema
class GetTimeSchema(BaseModel):
    """Returns current date, current time or both."""
    kind: Optional[str] = Field(description='kind of information to retrieve "date", "time" or "both"')
    
# definition using @tool decorator
@tool(args_schema=GetTimeSchema)
def get_time(kind: str = "both") -> str:
    """Returns current date, current time or both.

    Args:
        kind: date, time or both
    """
    if kind == "date":
        date = datetime.now().strftime("%d/%m/%Y")
        return f"Current date: {date}"
    elif kind == "time":
        time = datetime.now().strftime("%H:%M:%S")
        return f"Current time: {time}"
    else:
        date = datetime.now().strftime("%d/%m/%Y")
        time = datetime.now().strftime("%H:%M:%S")
        return f"Current date: {date}, Current time: {time}"

## Calculator Tool

class CalculatorSchema(BaseModel):
    """allow calculation of only basic operations: + - * and /
    with a string input expression """
    expression: str = Field(..., description="expression to calculate, example '12 * 3'")
    
#function to use in the tool
def calculator(expression: str ) -> Union[str, int, float]:
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
        '/': operator.truediv
    }
    tokens = re.findall(r'\d+\.?\d*|\+|\-|\*|\/|÷|x|X', expression)
    
    if not tokens:
        raise ToolException(f"Invalid expression '{expression}', should only contain one of the following operators + - * and /")
    
    current_value = float(tokens.pop(0))
    
    while tokens:
        # The next token should be an operator
        op = tokens.pop(0)
        
        # The next token should be a number
        if not tokens:
            raise ToolException(f"Incomplete expression '{expression}'")
        try:
            next_value = float(tokens.pop(0))
            
        except ValueError:
            raise ToolException("Invalid number format")
        
        except:
            raise ToolException("Invalid operation")
        
        # check division by 0
        if op in ['/','÷'] and next_value == 0:
            raise ToolException("cannot divide by 0")
        
        current_value = ops[op](current_value, next_value)
    
    return current_value


# tool error handler
def _handle_error(error: ToolException) -> str:
    return f"The following errors occurred during Calculator tool execution: `{error.args}`"


# tool definition
calculator = StructuredTool.from_function(
    func=calculator,
    args_schema=CalculatorSchema,
    handle_tool_error=_handle_error,#True,
)

## Python repl

# tool schema
class ReplSchema(BaseModel):
    "A Python shell. Use this to execute python commands. Input should be a valid python commands and expressions. If you want to see the output of a value, you should print it out with `print(...), if you need a specific module you should import it`."
    command: str = Field(..., description="python code to execute to evaluate")
    
# tool definition
python_repl = PythonREPL()
python_repl = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
    args_schema=ReplSchema
)