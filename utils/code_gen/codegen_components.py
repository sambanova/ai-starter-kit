import os
import sys
from typing import List, Any, Dict
from utils.rag.base_components import BaseComponents
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt
from langchain_experimental.utilities import PythonREPL
import re

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

class CodeGenComponents(BaseComponents):

    def python_parser(self, text: str) -> str:
        """
        Parse the given text to extract Python code blocks.

        Args:
            text (str): The input text to parse.

        Returns:
            str: The extracted Python code blocks, or the original text if no code blocks are found.
        """
         
        # text = text.content # For Ollama only
        code_blocks = re.findall(r"```python\s+(.*?)\s+```", text, re.S)

        if code_blocks:
            merged_code = "\n\n".join(code_blocks)
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
            self (object): The object instance.

        Returns:
            None
        """

        code_router_prompt: Any = load_prompt(repo_dir + "/" + self.prompts_paths["code_router_prompt"]) 
        self.code_router: Any = code_router_prompt | self.llm | JsonOutputParser()

    def init_codegen_chain(self) -> None:
        """
        Initializes the code generation chain by loading the codegen prompt and combining it with the LLM and Python parser.

        Args:
            self (object): The object instance.

        Returns: 
            None
        """
        
        codegen_prompt: Any = load_prompt(repo_dir + "/" + self.prompts_paths["codegen_prompt"]) 
        self.codegen: Any = codegen_prompt | self.llm | self.python_parser

    def init_codegen_qc_chain(self) -> None:
        """
        This class represents a codegen QC chain.

        Args:
            self (object): The object instance.

        Returns: 
            None
        """

        codegen_qc_prompt: Any = load_prompt(repo_dir + "/" + self.prompts_paths["codegen_qc_prompt"]) 
        self.codegen_qc: Any = codegen_qc_prompt | self.llm | JsonOutputParser()

    def init_refactor_chain(self) -> None:
        """
        Initializes the refactor chain by loading the refactor prompt and combining it with the LLM and Python parser.

        Args:
            self (object): The object instance.

        Returns: 
            None
        """
        
        refactor_prompt: Any = load_prompt(repo_dir + "/" + self.prompts_paths["refactor_prompt"]) 
        self.refactor: Any = refactor_prompt | self.llm | self.python_parser

    def init_failure_chain(self) -> None:

        failure_prompt = load_prompt(repo_dir + "/" + self.prompts_paths["failure_prompt"])
        self.failure_chain = failure_prompt | self.llm | StrOutputParser()

    def initialize_codegen(state: Dict) -> Dict:
        """
        Initializes the code generation process by creating a counter.

        Args:
            state (Dict): The current state of the code generation process.

        Returns:
            Dict: A dictionary containing the initialized counter.
        """

        print("Initializing counter and dataframe")

        return {"code_counter": 0, "rag_counter": 0}
        
    def route_question_to_code(self, state):
        """
        Route question to llm chain or code chain.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question: str = state["original_question"]
        print(question)
        source: Dict[str, str] = self.code_router.invoke({"question": question})  
        print(source['answer_type'])

        if source['answer_type'] == 'llm':
            print("---ROUTE QUESTION TO LLM---")
            return "llm"
        
        elif source['answer_type'] == 'codegen':
            print("---ROUTE QUESTION TO CODEGEN---")
            return "codegen"
        
    def pass_to_codegen(self, state: Dict) -> Dict:
        """
        Get a new query based on the given state.

        Args:
            state (Dict): A dictionary containing the current state.

        Returns:
            Dict: A dictionary containing the new query.
        """

        question: str = state["original_question"]

        return {"original_question": question}
        
    def code_generation(self, state: Dict) -> Dict:
        """
        Generates code based on the given question.

        Args:
            state (Dict[str, str]): A dictionary containing the question to generate code for.

        Returns:
            Dict[str, str]: A dictionary containing the question and the generated code.
        """

        print("---GENERATING CODE---")

        question: str = state["original_question"]
        answers: str = state["answers"]

        print(question)

        try:
            code = self.codegen.invoke({"question": question, "intermediate_answers": answers})
            print("Generated code: \n", code)
        except Exception as e:
            print(e)

        return {"original_question": question, "code": code}
        
    def determine_runnable_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function runs the given code and checks if it's runnable.

        Args:
            state (Dict[str, Any]): A dictionary containing the code to be run.

        Returns:
            Dict[str, Any]: A dictionary containing the result of the code run and a boolean indicating if the code is runnable.
        """

        code = state["code"]

        print("---QCING CODE---")

        python_repl = PythonREPL()
        try:
            result: Any = python_repl.run(code, timeout=30)
            print(result)
        except Exception as e:
            result: Any = e
            print(result)
        result = result.replace("}","").replace("{","")       
        output: Dict[str, Any] = self.codegen_qc.invoke({"output": result})
        print(output)
        is_runnable = output["runnable"]
        print("Code status: ", is_runnable)

        return {"runnable": is_runnable, "generation": result, "code": code}
    
    def decide_to_refactor(self, state: Dict[str, str]) -> str:
        """
        Determine whether to refactor based on the given state.

        Args:
            state (Dict[str, str]): A dictionary containing the current state of the system.

        Returns:
            str: A string indicating the result of the decision. Possible values are:
                - "unsuccessful": Refactoring is not needed.
                - "executed": The code is executable.
                - "exception": An exception occurred and refactoring is needed.
        """

        print("---DETERMINING IF REFACTOR IS NEEDED---")
        print("Attempt: ", state["code_counter"])
        runnable: str = state["runnable"]
        print(runnable)

        if int(state["code_counter"]) >= 3:
            return "unsuccessful"
        
        elif runnable == "executed":
            print("---EXECUTABLE---")
            return "executed"
        
        elif runnable == "exception":
            print("--ROUTING TO CODE REFACTOR---")
            return "exception"
        
    def refactor_code(self, state: Dict) -> Dict:
        """
        Refactor the given code.

        Args:
            state (Dict[str, Any]): A dictionary containing the code, error, and counter.

        Returns:
            Dict[str, Any]: A dictionary containing the refactored code, generation result, and counter.
        """

        print("--REFACTORING CODE---")

        code: str = state["code"]
        print("---CODE TO REFACTOR---")
        print(code)
        error: Any = state["error"]
        counter: int = state["code_counter"]
        counter += 1
        print("updated codegen counter: ", counter)

        refactor: str = self.refactor.invoke({"code": code, "error": error})
        print("---CODE--- \n", code)
        print("---REFACTORED CODE--- \n")
        print(refactor)
        python_repl = PythonREPL()
        print("---TESTING CODE---")
        try:
            result: Any = python_repl.run(refactor, timeout=30)
            return {"code": refactor, "generation": result, "code_counter": counter}
        except Exception as e:
            print(e)
            result: Any = e
            return {"code": refactor, "error": result, "code_counter": counter}
        
    def code_error_msg(self, state: Dict):
        """
        Generate an error message for code generation.

        Args:
            state (Dict[str, str]): A dictionary containing the code and error information.

        Returns:
            Dict[str, str]: A dictionary containing the error message.
        """

        code = state["code"]
        error = state["error"]
        answers = state["answers"]

        msg = f"""I'm sorry, but I cannot fix the code at this time.  Here is as far as I could get:
        {code}.\n\n  The underlying error is: {error}"""

        if not isinstance(answers, List):
            answers = [answers]
        answers.append(msg)

        return {"answers": answers}