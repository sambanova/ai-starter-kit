import json
import operator
import os
import re
import sys
from datetime import datetime
from pprint import pprint
from typing import Optional, Union

import yaml
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, Tool, ToolException, tool
from langchain_experimental.utilities import PythonREPL

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.sambanova_endpoint import SambaStudioFastCoE
from vectordb.vector_db import VectorDb  # type: ignore

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

load_dotenv(os.path.join(repo_dir, '.env'))


## Get configs for tools
def get_config_info(config_path: str) -> dict:
    """
    Loads json config file
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    tools_info = config['tools']

    return tools_info


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


@tool(args_schema=QueryDBSchema)
def query_db(query: str) -> str:
    """query generation tool. Use this to generate sql queries and retrieve the results from a database.
    Do not pass sql queries directly. Input must be a natural language question or instruction."""

    # get tool configs
    query_db_info = get_config_info(CONFIG_PATH)['query_db']

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
    elif query_db_info['llm']['api'] == 'fastcoe':
        llm = SambaStudioFastCoE(
            max_tokens=query_db_info['llm']['max_tokens_to_generate'],
            model=query_db_info['llm']['select_expert'],
        )
    else:
        raise ValueError(
            f"Invalid LLM API: {query_db_info['llm']['api']}, only 'sambastudio', 'sambaverse' and 'fastcoe' are supported."
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


## translation tool
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

    # get tool configs
    translate_info = get_config_info(CONFIG_PATH)['translate']

    # set the llm based in tool configs
    if translate_info['llm']['api'] == 'sambastudio':
        if translate_info['llm']['coe']:
            llm = SambaStudio(
                streaming=True,
                model_kwargs={
                    'max_tokens_to_generate': translate_info['llm']['max_tokens_to_generate'],
                    'select_expert': translate_info['llm']['select_expert'],
                    'temperature': translate_info['llm']['temperature'],
                },
            )
        else:
            llm = SambaStudio(
                model_kwargs={
                    'max_tokens_to_generate': translate_info['llm']['max_tokens_to_generate'],
                    'temperature': translate_info['llm']['temperature'],
                },
            )
    elif translate_info['llm']['api'] == 'sambaverse':
        llm = Sambaverse(  # type:ignore
            sambaverse_model_name=translate_info['llm']['sambaverse_model_name'],
            model_kwargs={
                'max_tokens_to_generate': translate_info['llm']['max_tokens_to_generate'],
                'select_expert': translate_info['llm']['select_expert'],
                'temperature': translate_info['llm']['temperature'],
            },
        )
    elif translate_info['llm']['api'] == 'fastcoe':
        llm = SambaStudioFastCoE(
            max_tokens=translate_info['llm']['max_tokens_to_generate'],
            model=translate_info['llm']['select_expert'],
        )
    else:
        raise ValueError(
            f"Invalid LLM API: {translate_info['llm']['api']}, only 'sambastudio', 'fastcoe' and'sambaverse' are supported."
        )

    return llm.invoke(f'Translate from {origin_language} to {final_language}: {input_sentence}')


## RAG tool
# tool schema


class RAGSchema(BaseModel):
    """Returns information from a document knowledge base"""

    query: str = Field(description='input question to solve using the knowledge base')


@tool(args_schema=RAGSchema)
def rag(query: str) -> str:
    """Returns information from a document knowledge base

    Args:
        query: str = input question to solve using the knowledge base
    """

    # get tool configs
    rag_info = get_config_info(CONFIG_PATH)['rag']

    # set the llm based in tool configs
    if rag_info['llm']['api'] == 'sambastudio':
        if rag_info['llm']['coe']:
            # Using SambaStudio CoE expert as model for generating the SQL Query
            llm = SambaStudio(
                streaming=True,
                model_kwargs={
                    'max_tokens_to_generate': rag_info['llm']['max_tokens_to_generate'],
                    'select_expert': rag_info['llm']['select_expert'],
                    'temperature': rag_info['llm']['temperature'],
                },
            )
        else:
            # Using SambaStudio endpoint as model for generating the SQL Query
            llm = SambaStudio(
                model_kwargs={
                    'max_tokens_to_generate': rag_info['llm']['max_tokens_to_generate'],
                    'temperature': rag_info['llm']['temperature'],
                },
            )
    elif rag_info['llm']['api'] == 'sambaverse':
        # Using Sambaverse expert as model for generating the SQL Query
        llm = Sambaverse(  # type:ignore
            sambaverse_model_name=rag_info['llm']['sambaverse_model_name'],
            model_kwargs={
                'max_tokens_to_generate': rag_info['llm']['max_tokens_to_generate'],
                'select_expert': rag_info['llm']['select_expert'],
                'temperature': rag_info['llm']['temperature'],
            },
        )
    elif rag_info['llm']['api'] == 'fastcoe':
        llm = SambaStudioFastCoE(
            max_tokens=rag_info['llm']['max_tokens_to_generate'],
            model=rag_info['llm']['select_expert'],
        )
    else:
        raise ValueError(
            f"Invalid LLM API: {rag_info['llm']['api']}, only 'sambastudio' and'sambaverse' are supported."
        )

    vdb = VectorDb()

    # load embedding model
    embeddings = vdb.load_embedding_model(
        type=rag_info['embedding_model']['type'],
        batch_size=rag_info['embedding_model']['batch_size'],
        coe=rag_info['embedding_model']['coe'],
        select_expert=rag_info['embedding_model']['select_expert'],
    )

    # set vectorstore and retriever
    vectorstore = vdb.load_vdb(os.path.join(kit_dir, rag_info['vector_db']['path']), embeddings, db_type='chroma')
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'score_threshold': rag_info['retrieval']['score_treshold'],
            'k': rag_info['retrieval']['k_retrieved_documents'],
        },
    )
    #  qa_chain definition
    prompt = (
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
        'You are an assistant for question-answering tasks.\n'
        'Use the following pieces of retrieved contexts to answer the question. '
        'If the information that is relevant to answering the question does not appear in the retrieved contexts, '
        'say "Could not find information.". Provide a concise answer to the question. '
        'Do not provide any information that is not asked for in the question. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n'
        'Question: {question} \n'
        'Context: {context} \n'
        '\n ------- \n'
        'Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    )
    retrieval_qa_prompt = PromptTemplate.from_template(prompt)
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
    source_documents = set([json.loads(doc.metadata['data_source'])['url'] for doc in response['source_documents']])

    return f'Answer: {answer}\nSource Document(s): {str(source_documents)}'
