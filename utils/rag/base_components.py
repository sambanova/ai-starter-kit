import os
import yaml
import torch
import base64
import nest_asyncio
from typing import List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from IPython.display import display, HTML
from langchain_community.embeddings.sambanova import SambaStudioEmbeddings
from langchain_core.prompts import load_prompt
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod
from langchain_community.llms.sambanova import SambaStudio, Sambaverse

class BaseComponents:

    def __init__(self, configs) -> None:

        self.configs = configs
        self.prompts_paths = configs["prompts"]
        
    @staticmethod
    def load_config(filename: str) -> dict:
        """
        Loads a YAML configuration file and returns its contents as a dictionary.

        Args:
            filename (str): The path to the YAML configuration file.

        Returns:
            Dict: A dictionary containing the configuration file's contents.
        """

        try:
            with open(filename, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The YAML configuration file {filename} was not found.") 
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")

    def load_embedding_model(self):
        """
        Loads an embedding model.

        Args:
            None

        Returns:
            Any: The loaded embedding model.
        """

        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info) 

        return embeddings  

    def init_llm(self):
        """
        Initializes the Large Language Model (LLM) based on the specified API.

        Args:
            self: The instance of the class.

        Returns:
            The self.llm attribute.
        """
        
        if self.configs["api"] == "sambaverse":
            self.llm = Sambaverse(
                    sambaverse_model_name=self.configs["llm"]["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs= {
                        "do_sample": False, 
                        "max_tokens_to_generate": self.configs["llm"]["max_tokens_to_generate"],
                        "temperature": self.configs["llm"]["temperature"],
                        "process_prompt": True,
                        "select_expert": self.configs["llm"]["sambaverse_select_expert"]
                    }
                )
            
        elif self.configs["api"] == "sambastudio":
            self.llm = SambaStudio(
                streaming=True,
                model_kwargs={
                    "do_sample": False,
                    "max_tokens_to_generate": self.configs["llm"]["max_tokens_to_generate"],
                    "process_prompt": False,
                    "select_expert": self.configs["llm"]["sambaverse_select_expert"],
                }
            )

            # local_llm = 'llama3'
            # self.llm = ChatOllama(model=local_llm, temperature=0)

    def _format_docs(self, docs: List) -> str:
        """
        Formats the page content of a list of documents into a single string.

        Args:
            docs (List): A list of Langchain Document objects.

        Returns:
            str: A string containing the formatted page content of the documents.
        """

        return "\n\n".join(doc.page_content for doc in docs)

    def init_embeddings(self):
        """
        Initializes the embeddings for the model.

        This method creates an instance of SambaStudioEmbeddings and assigns it to the self.embeddings attribute.

        Args:
            None

        Returns:
            None
        """

        self.embeddings = SambaStudioEmbeddings()
    
    def _display_image(self, image_bytes: bytes, width=512):
        """
        Displays an image from a byte string.

        Args:
            image_bytes (bytes): The byte string representing the image.
            width (int, optional): The width of the displayed image. Defaults to 512.

        Returns:
            The displayed image, via HTML.
        """

        decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
        html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
        display(HTML(html))
    
    def display_graph(self, app):
        """
        Displays a graph using Mermaid.

        This method uses the Mermaid library to generate a graph and then displays it.

        Args:
            app: The application object that contains the graph to be displayed.

        Returns:
            The populated self._display_image attribute.
        """

        nest_asyncio.apply() # Required for Jupyter Notebook to run async functions

        img_bytes = app.get_graph().draw_mermaid_png(
        curve_style=CurveStyle.LINEAR,
        node_colors=NodeColors(start="#ffdfba", end="#baffc9", other="#fad7de"),
        wrap_label_n_words=9,
        output_file_path=None,
        draw_method=MermaidDrawMethod.PYPPETEER,
        background_color="white",
        padding=10
)

        self._display_image(img_bytes)

    def init_base_llm_chain(self):
        """
        Initializes the base LLM chain.

        This method loads the base LLM prompt and combines it with the LLM model
        and a StrOutputParser to create the base LLM chain.

        Args:
            self (object): The object instance.

        Returns:
            The self.base_llm_chain attribute.
        """

        base_llm_prompt = load_prompt(self.confis["prompts"]["base_llm_prompt"])
        self.base_llm_chain = base_llm_prompt| self.llm | StrOutputParser()

    def rerank_docs(self, query, docs, final_k):
        """
        Rerank a list of documents based on their relevance to a given query.

        This method uses a pre-trained reranker model to compute the relevance scores of the documents to the query, and then returns the top-scoring documents.

        Args:
            query (str): The query string.
            docs (list): A list of Langchain document objects, each containing a page_content attribute.
            final_k (int): The number of top-scoring documents to return.

        Returns:
            list: A list of the top-scoring documents, in order of their relevance to the query.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.configs["retrieval"]["reranker"])
        reranker = AutoModelForSequenceClassification.from_pretrained(self.configs["retrieval"]["reranker"])
        pairs = []
        for d in docs:
            pairs.append([query, d.page_content])

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
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
        scores_sorted_idx = sorted(
            range(len(scores_list)), key=lambda k: scores_list[k], reverse=True
        )

        docs_sorted = [docs[k] for k in scores_sorted_idx]
        # docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
        docs_sorted = docs_sorted[:final_k]

        return docs_sorted

    def llm_generation(self, state: Dict) -> Dict[str, str]:
        """
        Generates a response based on the internal knowledge.

        Args:
            state (Dict): The current state dict of the app.

        Returns:
            Dict: The updated state dict of the app.
        """
        
        print("---GENERATING FROM INTERNAL KNOWLEDGE---")
        question: str = state["question"]
        
        # RAG generation
        generation: str = self.base_llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}