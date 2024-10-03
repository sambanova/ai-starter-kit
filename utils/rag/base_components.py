import base64
import os
import sys
from typing import Any, List

import nest_asyncio  # type: ignore
import torch
import yaml  # type: ignore
from IPython.display import HTML, display
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langgraph.graph.state import CompiledStateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.logging_utils import log_method  # type: ignore
from utils.model_wrappers.api_gateway import APIGateway


class BaseComponents:
    def __init__(self, configs: dict) -> None:
        self.configs = configs

    @staticmethod
    def load_config(filename: str) -> dict:
        """
        Loads a YAML configuration file and returns its contents as a dictionary.

        Args:
            filename: The path to the YAML configuration file.

        Returns:
            A dictionary containing the configuration file's contents.
        """

        try:
            with open(filename, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The YAML configuration file {filename} was not found.')
        except yaml.YAMLError as e:
            raise RuntimeError(f'Error parsing YAML file: {e}')

    def init_llm(self) -> None:
        """
        Initializes the Large Language Model (LLM) based on the specified API.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        self.llm = APIGateway.load_llm(
            type=self.configs['api'],
            streaming=True,
            coe=self.configs['llm']['coe'],
            do_sample=self.configs['llm']['do_sample'],
            max_tokens_to_generate=self.configs['llm']['max_tokens_to_generate'],
            temperature=self.configs['llm']['temperature'],
            select_expert=self.configs['llm']['select_expert'],
            process_prompt=False,
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formats the page content of a list of documents into a single string.

        Args:
            docs: A list of Langchain Document objects.

        Returns:
            A string containing the formatted page content of the documents.
        """

        return '\n\n'.join(doc.page_content for doc in docs)

    def init_embeddings(self) -> None:
        """
        Initializes the embeddings for the model.

        This method creates an instance of SambaStudio Embeddings or E5 Large from HuggingFaceInstructEmbeddings
        to be run on cpu and assigns it to the self.embeddings attribute.

        Args:
            None

        Returns:
            None
        """

        self.embeddings = APIGateway.load_embedding_model(
            type=self.configs['embedding_model']['type'],
            batch_size=self.configs['embedding_model']['batch_size'],
            coe=self.configs['embedding_model']['coe'],
            select_expert=self.configs['embedding_model']['select_expert'],
        )

    def _display_image(self, image_bytes: bytes, width: int = 512) -> None:
        """
        Displays an image from a byte string.

        Args:
            image_bytes: The byte string representing the image.
            width: The width of the displayed image. Defaults to 512.

        Returns:
            None
        """

        decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
        html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
        display(HTML(html))

    def display_graph(self, app: CompiledStateGraph) -> None:
        """
        Prepares img_bytes of a graph using Mermaid for display purposes.
        Will be passed to the _display_image method for actual display.

        This method uses the Mermaid library to generate a graph and then displays it.

        Args:
            app: The application object that contains the graph to be displayed.

        Returns:
            None
        """

        nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

        img_bytes = app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color='white',
            padding=10,
        )

        self._display_image(img_bytes)

    def init_base_llm_chain(self) -> None:
        """
        Initializes the base LLM chain.

        This method loads the base LLM prompt and combines it with the LLM model
        and a StrOutputParser to create the base LLM chain.

        Args:
            None

        Returns:
            None
        """

        base_llm_prompt: Any = load_prompt(repo_dir + '/' + self.configs['prompts']['base_llm_prompt'])
        self.base_llm_chain = base_llm_prompt | self.llm | StrOutputParser()  # type: ignore

    def rerank_docs(self, query: str, docs: List[Document], final_k: int) -> List[Document]:
        """
        Rerank a list of documents based on their relevance to a given query.

        This method uses a pre-trained reranker model to compute the relevance scores of the documents
        to the query, and then returns the top-scoring documents.

        Args:
            query: The query string.
            docs: A list of Langchain document objects, each containing a page_content attribute.
            final_k: The number of top-scoring documents to return.

        Returns:
            A list of the top-scoring Lanchgain Documents, in order of their relevance to the query.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.configs['retrieval']['reranker'])
        reranker = AutoModelForSequenceClassification.from_pretrained(self.configs['retrieval']['reranker'])
        pairs = []
        for d in docs:
            pairs.append([query, d.page_content])

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            scores = (
                reranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        scores_list = scores.tolist()
        scores_sorted_idx = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)

        docs_sorted: List[Document] = [docs[k] for k in scores_sorted_idx]
        # docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
        docs_sorted = docs_sorted[:final_k]

        return docs_sorted

    @log_method
    def llm_generation(self, state: dict) -> dict:
        """
        Generates a response from the selected LLM (without any external knowledge or context).

        Args:
            state: The current state dict of the app.

        Returns:
            dict: The updated state dict of the app.
        """

        print('---GENERATING FROM INTERNAL KNOWLEDGE---')
        question: str = state['question']

        # RAG generation
        generation: str = self.base_llm_chain.invoke({'question': question})
        return {'question': question, 'generation': generation}
