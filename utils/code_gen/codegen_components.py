import os
import sys
from typing import List, Any, Dict
from utils.rag.base_components import BaseComponents  # type: ignore
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import load_prompt
from langchain_experimental.utilities import PythonREPL
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.logging_utils import log_method # type: ignore

class CodeGenComponents(BaseComponents):
    def python_parser(self, text: str) -> str:
        """
        Parse the given text to extract Python code blocks.  If Python code is parsed based on
        the regex pattern, it will return the merged code blocks.
        Otherwise, it will return the original text.

        Args:
            text: The input text to parse.

        Returns:
            The extracted Python code blocks, or the original text if no code blocks are found.
        """

        code_blocks = re.findall(r'```python\s+(.*?)\s+```', text, re.S)

        if code_blocks:
            merged_code = '\n\n'.join(code_blocks)
            return merged_code
        else:
            return text

    def init_code_router(self) -> None:
        """
        Initializes the code router with the loaded prompt and LLM.

        This method loads the code router prompt from the prompts_paths dictionary
        and combines it with the LLM (Large Language Model) to create the code router.
        The resulting code router is then parsed into a JSON output.

        Args:
            None

        Returns:
            None
        """

        code_router_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['code_router_prompt'])
        self.code_router = code_router_prompt | self.llm | JsonOutputParser()

    def init_codegen_chain(self) -> None:
        """
        Initializes the code generation chain by loading the codegen prompt and
        combining it with the LLM and Python parser.

        Args:
            None

        Returns:
            None
        """

        codegen_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['codegen_prompt'])
        self.codegen = codegen_prompt | self.llm | self.python_parser

    def init_codegen_qc_chain(self) -> None:
        """
        This method initializes the codegen QC chain.

        Args:
            None

        Returns:
            None
        """

        codegen_qc_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['codegen_qc_prompt'])
        self.codegen_qc = codegen_qc_prompt | self.llm | JsonOutputParser()

    def init_refactor_chain(self) -> None:
        """
        Initializes the refactor chain by loading the refactor prompt and
        combining it with the LLM and Python parser.

        Args:
            None

        Returns:
            None
        """

        refactor_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['refactor_prompt'])
        self.refactor = refactor_prompt | self.llm | self.python_parser

    # TODO: move to rag?
    def init_failure_chain(self) -> None:
        """
        Initializes the failure chain by loading the failure prompt and
        combining it with the language model and a string output parser.

        Args:
            None

        Returns:
            None
        """

        failure_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['failure_prompt'])
        self.failure_chain = failure_prompt | self.llm | StrOutputParser()

    @log_method
    def initialize_codegen(self, state: dict) -> dict:
        """
        Initializes the code generation state and the required counters.

        Args:
            state (Dict): The current state of the code generation process.

        Returns:
            Dict: The state dictionary containing the initialized counters.
        """

        print('Initializing counter and dataframe')

        return {'code_counter': 0, 'rag_counter': 0}

    @log_method
    def route_question_to_code(self, state: dict) -> str:
        """
        Route question to llm chain or code chain.

        Args:
            state: The current graph state.

        Returns:
            Next node to call, llm or codegen.
        """

        print('---ROUTE QUESTION---')
        question: str = state['original_question']
        print(question)
        source: Dict[str, str] = self.code_router.invoke({'question': question})
        print(source['answer_type'])

        if source['answer_type'] == 'llm':
            print('---ROUTE QUESTION TO LLM---')
            routing = 'llm'

        elif source['answer_type'] == 'codegen':
            print('---ROUTE QUESTION TO CODEGEN---')
            routing = 'codegen'

        return routing

    @log_method
    def pass_to_codegen(self, state: dict) -> dict:
        """
        Get a new query based on the given state.

        Args:
            state: A dictionary containing the current state.

        Returns:
            The state updated dictionary containing the new query.
        """

        question: str = state['original_question']

        return {'original_question': question}

    @log_method
    def code_generation(self, state: dict) -> dict:
        """
        Generates code based on the given question.

        Args:
            state: The state dictionary containing the question
            to generate code for.

        Returns:
            The updated state dictionary containing the generated code.
        """

        print('---GENERATING CODE---')

        question: str = state['original_question']
        answers: str = state['answers']

        print(question)

        try:
            code = self.codegen.invoke({'question': question, 'intermediate_answers': answers})
            print('Generated code: \n', code)
        except Exception as e:
            print(e)

        return {'original_question': question, 'code': code}

    @log_method
    def determine_runnable_code(self, state: dict) -> dict:
        """
        This function runs the given code and checks if it's runnable.
        WARNING - this could be potentially dangerous.
        From: https://python.langchain.com/v0.2/docs/integrations/tools/python/

        "Python REPL can execute arbitrary code on the host machine
        (e.g., delete files, make network requests). Use with caution.

        For more information general security guidelines,
        please see https://python.langchain.com/v0.2/docs/security/."

        Args:
            state: The current state dictionary.

        Returns:
            The updated state dictionary containing the result of the
            code run and a boolean indicating if the code is runnable.
        """

        code: str = state['code']

        print('---QCING CODE---')

        python_repl = PythonREPL()
        # Attempt to run the generated code
        try:
            result: str = python_repl.run(code, timeout=30)
            print(result)
        # If an exception is encountered, capture the exception and pass it
        # as the result, so it can be later refactored.
        except Exception as e:
            result = str(e)
            print(result)
        # Tidy up curly brackets that will throw exceptions when invoking the next chain.
        result = result.replace('}', '').replace('{', '')
        output: Dict[str, Any] = self.codegen_qc.invoke({'output': result})
        print(output)
        is_runnable = output['runnable']
        print('Code status: ', is_runnable)

        return {'runnable': is_runnable, 'generation': result, 'code': code}

    @log_method
    def decide_to_refactor(self, state: dict) -> str:
        """
        Determine whether to refactor based on the given state.

        Args:
            state: A dictionary containing the current state of the system.

        Returns:
            A string indicating the result of the decision. Possible values are:
                - "unsuccessful": Refactoring is not needed.
                - "executed": The code is executable.
                - "exception": An exception occurred and refactoring is needed.
        """

        print('---DETERMINING IF REFACTOR IS NEEDED---')
        print('Attempt: ', state['code_counter'])
        runnable: str = state['runnable']
        print(runnable)

        if int(state['code_counter']) >= self.configs['codegen']['max_attemps']:
            routing = 'unsuccessful'

        elif runnable == 'executed':
            print('---EXECUTABLE---')
            routing = 'executed'

        elif runnable == 'exception':
            print('--ROUTING TO CODE REFACTOR---')
            routing = 'exception'

        return routing

    @log_method
    def refactor_code(self, state: dict) -> dict:
        """
        Refactor the given code.

        Args:
            state: The current state dictionary containing the code, error,
            and counter, and other variables.

        Returns:
            The updated state dictionary containing the refactored code,
            generation result, and counter.
        """

        print('--REFACTORING CODE---')

        code: str = state['code']
        print('---CODE TO REFACTOR---')
        print(code)
        error: str = state['error']
        counter: int = state['code_counter']
        counter += 1
        print('updated codegen counter: ', counter)

        refactor: str = self.refactor.invoke({'code': code, 'error': error})
        result: str = ''

        print('---CODE--- \n', code)
        print('---REFACTORED CODE--- \n')
        print(refactor)
        python_repl = PythonREPL()
        print('---TESTING CODE---')
        try:
            result = python_repl.run(refactor, timeout=30)
            return {'code': refactor, 'generation': result, 'code_counter': counter}
        except Exception as e:
            print(e)
            result = str(e)
            return {'code': refactor, 'error': result, 'code_counter': counter}

    @log_method
    def code_error_msg(self, state: dict) -> dict:
        """
        Generate an error message for code generation.

        Args:
            state: The current state dictionary containing the code and error information.

        Returns:
            The updated dictionary containing the error message.
        """

        code: str = state['code']
        error: str = state['error']
        answers: List[str] = state['answers']

        msg = f"""I'm sorry, but I cannot fix the code at this time.  Here is as far as I could get:
        {code}.\n\n  The underlying error is: {error}"""

        if not isinstance(answers, list):
            answers = [answers]
        answers.append(msg)

        return {'answers': answers}
