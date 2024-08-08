import os
import sys
from typing import Any, Dict, List, Optional
from utils.rag.base_components import BaseComponents  # type: ignore
from langchain_core.output_parsers import JsonOutputParser
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt
from langchain_core.embeddings import Embeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


class SupervisorComponents(BaseComponents):
    """
    This class represents the components of the Supervisor for RAG.
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

    def init_supervisor_router(self) -> None:
        """
        Initializes the supervisor router.
        This method loads the supervisor router prompt from the
        repository and combines it with the language model and a
        JSON output parser.
        """

        supervisor_router_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['supervisor_prompt'])
        self.supervisor = supervisor_router_prompt | self.llm | JsonOutputParser()

    def supervisor_router(self, state: dict) -> dict:
        """
        This method is the supervisor router, which handles the state of the conversation.
        
        Args:
            state: A dictionary containing the current state
            of the conversation, including the question, question history,
            answer history, and teams.
        Returns:
            The state dictionary with the updated next step in the conversation,
            as determined by the supervisor.
        """

        question: str = state['question']
        question_history: List[str] = state['query_history']
        answer_history: List[str] = state['answer_history']

        print('---SUPERVISOR ROUTER INPUTS---')
        print(question)
        print(question_history)
        print(answer_history)

        response: Dict[str, str] = self.supervisor.invoke(
            {
                'question_history': question_history,
                'answer_history': answer_history,
                'question': question,
                'state': state,
            }
        )

        print('---NEXT ACTION---')
        print(response['next'])

        return {'next': response['next']}
