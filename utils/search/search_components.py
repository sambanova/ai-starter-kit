import os
import sys
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.embeddings import Embeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.logging_utils import log_method  # type: ignore
from utils.rag.base_components import BaseComponents  # type: ignore


class SearchComponents(BaseComponents):
    """
    This class represents the components of the Search system.

    Attributes:
    - qa_chain: The QA chain component of the RAG model.
    - vectorstore: The vector store component of the RAG model.
    - embeddings: The embeddings component of the RAG model.
    - examples: Optional examples dictionary.
    - configs: The configuration dictionary.
    - prompts_paths (Dict): The paths to the prompts in the configuration dictionary.
    """

    def __init__(
        self, configs: str, embeddings: Embeddings, vectorstore: Chroma, examples: Optional[Dict[Any, Any]] = None
    ) -> None:
        """
        Initializes the RAG components.

        Args:
            configs: The configuration file path.
            embeddings: The embeddings model.
            vectorstore: The vector store object.
            examples: The examples dictionary. Defaults to None.
        """

        self.qa_chain = None
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples: Optional[Dict] = examples

        self.configs: Dict = self.load_config(configs)
        self.prompts_paths: Dict = self.configs['prompts']

    @log_method
    def tavily_web_search(self, state: dict) -> dict:
        """
        Web search using Tavily based based on the question

        Args:
            state: The current graph state

        Returns:
            The state dictionary with appended web results to documents
        """

        web_search_tool = TavilySearchResults(max_results=self.configs['retrieval']['n_tavily_results'])

        print('---WEB SEARCH---')
        question: str = state['original_question']
        # Case in which the user asks for search prior to any
        if not question:
            question = state['question']
        search_strins = ['search', 'internet', 'web']
        for string in search_strins:
            if string in state['question'].lower():
                question = state['question']
        documents: List[Document] = state['documents']
        print(question)

        # Web search
        docs = web_search_tool.invoke({'query': question})
        try:
            web_results: str = '\n'.join([d['content'] for d in docs])
        except:
            web_results = docs
        try:
            sources: list = [d['url'] for d in docs]
        except:
            sources = ['']

        doc_web_results: Document = Document(page_content=web_results, metadata={'filename': sources})
        if documents is not None:
            documents.append(doc_web_results)
        else:
            documents = [doc_web_results]

        print(documents)
        return {'documents': documents, 'question': question}

    @log_method
    def final_answer_search(self, state: dict) -> dict:
        """
        This method is used to generate the final answer based on the original question and the generated text.

        Args:
            state: The state dictionary containing the original
            question, the generated text, and other variables.

        Returns:
            The updated state dictionary containing the final answer.
        """

        original_question: str = state['original_question']
        generation: str = state['generation']

        print('---Final Generation---')
        print(generation)

        final_answer: str = self.final_chain.invoke({'question': original_question, 'generation': generation})

        return {'generation': final_answer, 'original_question': ''}
