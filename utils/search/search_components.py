import os
import sys
from typing import Dict, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.vectorstores import Chroma

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.rag.base_components import BaseComponents

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

    def __init__(self, 
                configs: str, 
                embeddings: object, 
                vectorstore: Chroma, 
                examples: dict=None) -> None:
        """
        Initializes the RAG components.

        Args:
            configs: The configuration file path.
            embeddings: The embeddings model.
            vectorstore: The vector store object.
            examples: The examples dictionary. Defaults to None.

        Returns:
            None
        """
        
        self.qa_chain = None
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples: Optional[Dict] = examples
        
        self.configs: Dict = self.load_config(configs)
        self.prompts_paths: Dict = self.configs["prompts"]

    def tavily_web_search(self, state: dict) -> dict:
        """
        Web search using Tavily based based on the question

        Args:
            state: The current graph state

        Returns:
            The state dictionary with appended web results to documents
        """

        web_search_tool = TavilySearchResults(max_results=self.configs["retrieval"]["n_tavily_results"])

        print("---WEB SEARCH---")
        question: str = state["original_question"]
        # Case in which the user asks for search prior to any
        if not question:
            question = state["question"]
        search_strins =  ["search", "internet", "web"]
        for string in search_strins:
            if string in state["question"].lower():
                question = state["question"]
        documents: List[Document] = state["documents"]
        print(question)

        # Web search
        docs = web_search_tool.invoke({"query": question})
        try:
            web_results: str = "\n".join([d["content"] for d in docs])
        except: 
            web_results: str = docs
        try:
            sources: list = [d["url"] for d in docs]
        except: 
            sources: list = [""]
            
        web_results: List[Document] = Document(page_content=web_results, metadata={"filename": sources})
        if documents is not None:
            documents.append(web_results)
        else:
            documents: List[Document] = [web_results]

        print(documents)
        return {"documents": documents, "question": question}
