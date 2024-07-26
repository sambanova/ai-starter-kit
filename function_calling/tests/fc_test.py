#!/usr/bin/env python3
"""
Function Calling (FC) Test Script

This script tests the functionality of the Function Calling using unittest.

Usage:
    python tests/fc_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
from typing import Callable, Generator, Optional

current_dir = os.getcwd()
kit_dir = current_dir # absolute path for function_calling kit dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '..')) # absolute path for ai-starter-kit root repo

sys.path.append(kit_dir)
sys.path.append(repo_dir)

#load_dotenv(os.path.join(repo_dir, '.env'))

from function_calling.src.function_calling import FunctionCallingLlm  # type: ignore
from function_calling.src.tools import calculator, get_time, python_repl # type: ignore

print(get_time.invoke({'kind': 'both'}))

print(calculator.invoke('18*23.7 -5'))

print(python_repl.invoke({'command': 'for i in range(0,5):\n\tprint(i)'}))


tools = [get_time, calculator, python_repl]

fc = FunctionCallingLlm(tools) 

response = fc.function_call_llm('what time is it?', max_it=5, debug=False)
print(response)

response = fc.function_call_llm('it is time to go to sleep, how many hours last to 10pm?', max_it=5, debug=False)
print(response)

response = fc.function_call_llm(
    "sort this list of elements alphabetically ['screwdriver', 'pliers', 'hammer']", max_it=5, debug=False
)
print(response)