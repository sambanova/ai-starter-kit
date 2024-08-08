import os
import sys
from typing import Any, Dict, Optional
from utils.rag.base_components import BaseComponents  # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt
from langchain_core.embeddings import Embeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


class ReturnComponents(BaseComponents):
    """
    This class represents the components of the return message handler for RAG.

    Attributes:
    - qa_chain: The QA chain component of the RAG model.
    - vectorstore: The vector store component of the RAG model.
    - embeddings: The embeddings component of the RAG model.
    - examples: Optional examples dictionary.
    - configs: The configuration dictionary.
    - prompts_paths (Dict): The paths to the prompts in the configuration dictionary.
    """

    def __init__(
        self, configs: str, embeddings: Embeddings, vectorstore: Chroma, examples: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initializes the RAG components.

        Args:
            configs: The configuration file path.
            embeddings: The embeddings model.
            vectorstore: The vector store object.
            examples: The examples dictionary. Defaults to None.
        """

        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples: Optional[Dict[str, str]] = examples

        self.configs: Dict = self.load_config(configs)
        self.prompts_paths: Dict = self.configs['prompts']

    def init_return_message(self) -> None:
        """
        Initializes the return message by loading the prompt
        and combining it with the LLM and a string
        output parser.
        """

        return_message_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['return_message_prompt'])
        self.return_message_chain = return_message_prompt | self.llm | StrOutputParser()

    def return_message_to_user(self, state: dict) -> dict:
        """
        Returns a response to the user based on the given state.

        Args:
            state: The state dictionary containing the question, answer, and next state.

        Returns:
            dict: The state dictionary containing updated the response to the user.
        """

        question: str = state['question']
        answer: str = state['generation']
        next: str = state['next']

        response: str = self.return_message_chain.invoke({'question': question, 'answer': answer, 'next': next})
        print('---RESPONSE---')
        print(response)

        return {'generation': response}
