import functools
import logging
import multiprocessing
import operator
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from io import StringIO
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, Tool, ToolException, tool
from langchain_sambanova import ChatSambaNova, SambaNovaEmbeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
RAG_PROMPT = os.path.join(kit_dir, 'prompts', 'rag_tool.yaml')
QUERY_DB_PROMPT = os.path.join(kit_dir, 'prompts', 'query_db_tool.yaml')

load_dotenv(os.path.join(repo_dir, '.env'))


def load_chat_prompt(path: str) -> ChatPromptTemplate:
    """Load chat prompt from yaml file"""

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    config.pop('_type')

    template = config.pop('template')

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    messages = []
    if isinstance(template, str):
        messages.append(('human', template))

    elif isinstance(template, list):
        for item in template:
            messages.append((item['role'], item['content']))

    return ChatPromptTemplate(messages=messages, **config)


def get_config_info(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads json config file
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    tools_info: Dict[str, Any] = config['tools']

    return tools_info

@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")  # type: ignore[arg-type]
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")  # type: ignore[arg-type]

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, queue)
        # get the result from the worker function
        return queue.get()


class ToolClass(ABC):
    """Default class for creating configurable tools,
    that needs constants or parameters not sent in a tool calling event"""

    def __init__(self, config_path: str = CONFIG_PATH, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = get_config_info(config_path)

    @abstractmethod
    def get_tool(self) -> StructuredTool:
        pass


### Get time tool


# tool schema
class GetTimeSchema(BaseModel):
    """Returns current date, current time or both in dd/mm/yyyy format."""

    kind: Optional[str] = Field(
        description='kind of information to retrieve "date" in dd/mm/yyyy format, "time" or "both"'
    )


# definition using @tool decorator
@tool(args_schema=GetTimeSchema)
def get_time(kind: str = 'both') -> str:
    """Returns current date, current time or both in dd/mm/yyyy format.

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


### Calculator Tool


# tool schema
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


### Python standard shell, or REPL (Read-Eval-Print Loop)


# tool schema
class ReplSchema(BaseModel):
    "A Python shell. Use this to execute python commands. Input should be a valid python commands and expressions. If you want to see the output of a value, you should print it out with `print(...)`, if you need a specific module you should import it."  # noqa: E501

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


### SQL tool


class QueryDb(ToolClass):
    # tool schema
    class QueryDBSchema(BaseModel):
        "A query generation tool. Use this to generate sql queries and retrieve the results from a database. Do not pass sql queries directly. Input must be a natural language question or instruction."  # noqa: E501

        query: str = Field(..., description='natural language question or instruction.')

    def sql_finder(self, text: str) -> str:
        """Search in a string for a SQL query or code with format"""

        # regex for finding sql_code_pattern with format:
        # ```sql
        #    <query>
        # ```

        print(f'query_db: query generation LLM raw response: \n{text}\n')
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

    # tool definition
    def query_db(self, query: str) -> str:
        """query generation tool. Use this to generate sql queries and retrieve the results from a database.
        Do not pass sql queries directly. Input must be a natural language question or instruction."""

        # get tool configs
        query_db_info = self.config['query_db']

        llm = ChatSambaNova(api_key=self.kwargs.get('sambanova_api_key'), **query_db_info['llm'])

        if self.kwargs.get('session_temp_db') is not None:  # TODO pass this param
            db_path = self.kwargs.get('session_temp_db')
        else:
            db_path = os.path.join(kit_dir, query_db_info['db']['path'])
        db_uri = f'sqlite:///{db_path}'
        db = SQLDatabase.from_uri(db_uri)

        prompt = load_chat_prompt(QUERY_DB_PROMPT)

        # Chain that receives the natural language input and the table schema,
        # then pass the teh formatted prompt to the llm
        # and finally execute the sql finder method, retrieving only the filtered SQL query
        query_generation_chain = prompt | llm | StrOutputParser() | RunnableLambda(self.sql_finder)

        table_info = db.get_table_info()

        print(f'query_db: Calling query generation LLM with input: \n{query}\n')

        query = query_generation_chain.invoke({'input': query, 'table_info': table_info})

        print(f'query_db: query generation LLM filtered response: \n{query}\n')

        queries = query.split(';')

        query_executor = QuerySQLDataBaseTool(db=db)

        results = []
        for query in queries:
            if query.strip() != '':
                print(f'query_db: executing query: \n{query}\n')
                results.append(query_executor.invoke(query))
                print(f'query_db: query result: \n{results[-1]}\n')

        result = '\n'.join([f'Query {query} executed with result {result}' for query, result in zip(queries, results)])
        return result

    def get_tool(self) -> StructuredTool:
        tool = StructuredTool.from_function(
            func=self.query_db,
            name='query_db',
            args_schema=self.QueryDBSchema,
        )
        return tool


### Translation tool


class Translate(ToolClass):
    # tool schema
    class TranslateSchema(BaseModel):
        """Returns translated input sentence to desired language"""

        origin_language: str = Field(description='language of the original sentence')
        final_language: str = Field(description='language to translate the sentence into')
        input_sentence: str = Field(description='sentence to translate')

    # tool definition
    def translate(self, origin_language: str, final_language: str, input_sentence: str) -> str:
        """Returns translated input sentence to desired language

        Args:
            origin_language: language of the original sentence
            final_language: language to translate the sentence into
            input_sentence: sentence to translate
        """

        # get tool configs
        translate_info = self.config['translate']

        # set the llm based in tool configs
        llm = ChatSambaNova(api_key=self.kwargs.get('sambanova_api_key'), **translate_info['llm'])
        chain = llm | StrOutputParser()

        return chain.invoke(f'Translate from {origin_language} to {final_language}: {input_sentence}')

    def get_tool(self) -> StructuredTool:
        tool = StructuredTool.from_function(
            func=self.translate,
            name='translate',
            args_schema=self.TranslateSchema,
        )
        return tool


### RAG tool
class Rag(ToolClass):
    # tool schema
    class RAGSchema(BaseModel):
        """Returns information from a document knowledge base"""

        query: str = Field(description='input question to solve using the knowledge base')

    # tool definition
    def rag(self, query: str) -> str:
        """Returns information from a document knowledge base

        Args:
            query: str = input question to solve using the knowledge base
        """

        # get tool configs
        rag_info = self.config['rag']

        # set the llm based in tool configs
        llm = ChatSambaNova(api_key=self.kwargs.get('sambanova_api_key'), **rag_info['llm'])

        vdb = VectorDb()

        # load embedding model
        embeddings = SambaNovaEmbeddings(api_key=self.kwargs.get('sambanova_api_key'), **rag_info['embedding_model'])

        # set vectorstore and retriever
        vectorstore = vdb.load_vdb(
            os.path.join(kit_dir, rag_info['vector_db']['path']),
            embeddings,
            db_type='chroma',
            collection_name=rag_info['vector_db']['collection_name'],
        )
        retriever = vectorstore.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                'score_threshold': rag_info['retrieval']['score_treshold'],
                'k': rag_info['retrieval']['k_retrieved_documents'],
            },
        )
        #  qa_chain definition
        retrieval_qa_prompt = load_chat_prompt(RAG_PROMPT)
        qa_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            input_key='question',
            output_key='answer',
            prompt=retrieval_qa_prompt,
        )

        response = qa_chain.invoke({'question': query})
        answer = response['answer']

        source_documents = set([doc.metadata['filename'] for doc in response['source_documents']])

        return f'Answer: {answer}\nSource Document(s): {str(source_documents)}'

    def get_tool(self) -> StructuredTool:
        tool = StructuredTool.from_function(
            func=self.rag,
            name='rag',
            args_schema=self.RAGSchema,
        )
        return tool
