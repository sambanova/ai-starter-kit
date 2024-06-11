import os
import yaml
import torch
import base64
import nest_asyncio
from typing import Dict, List, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from IPython.display import display, HTML
from langchain_community.embeddings.sambanova import SambaStudioEmbeddings
from langchain_core.prompts import load_prompt
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod
from langchain_community.llms.sambanova import SambaStudio, Sambaverse

class BaseComponents:

    def __init__(self, configs: Dict[str, str]) -> None:
        """
        Base class for components that can be composed together.

        This class provides a basic structure for components that can be used to build more complex pipelines.
        """

        self.configs: Dict[str, str] = configs
        self.prompts_paths: str = configs["prompts"]
        
    @staticmethod
    def load_config(filename: str) -> Dict:
        """
        Loads a YAML configuration file and returns its contents as a dictionary.

        Args:
            filename (str): The path to the YAML configuration file.

        Returns:
            Dict: The contents of the YAML configuration file as a dictionary.
        """

        try:
            with open(filename, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The YAML configuration file {filename} was not found.") 
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")

    def load_embedding_model(self) -> Any:
        """
        Loads an embedding model from the vector database.

        Args:
            self (object): The object instance.

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
            None
        """
        
        if self.configs["api"] == "sambaverse":
            self.llm = Sambaverse(
                    sambaverse_model_name=self.configs["llm"]["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.configs["llm"]["max_tokens_to_generate"],
                        "temperature": self.configs["llm"]["temperature"],
                        "process_prompt": True,
                        "select_expert": self.configs["llm"]["sambaverse_select_expert"]
                    }
                )
            
        elif self.configs["api"] == "sambastudio":
            # self.llm = SambaStudio(
            #     streaming=True,
            #     model_kwargs={
            #             "max_tokens_to_generate": 2048,
            #             "do_sample": False,
            #             "temperature": 0.01,
            #             "select_expert": "Meta-Llama-3-70B-Instruct",
            #             "process_prompt": False,
            #         }
            # )
            local_llm = 'llama3'
            self.llm = ChatOllama(model=local_llm, temperature=0)

    def _format_docs(self, docs: List) -> str:
        """
        Formats the content of a list of documents into a single string.

        Args:
            docs (List): A list of Document objects.

        Returns:
            str: A string containing the formatted content of the documents.
        """

        return "\n\n".join(doc.page_content for doc in docs)

    def init_embeddings(self) -> None:
        """
        Initializes the embeddings for the component.

        Args:
            self (object): The object that this method is called on.

        Returns:
            None
        """

        self.embeddings: SambaStudioEmbeddings = SambaStudioEmbeddings()
    
    def _display_image(self, image_bytes: bytes, width: int = 512) -> None:
        """
        Displays an image from a bytes object.

        Args:
            image_bytes (bytes): The bytes object containing the image data.
            width (int, optional): The width of the displayed image. Defaults to 256.

        Returns:
            None
        """

        decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
        html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
        display(HTML(html))
    
    def display_graph(self, app) -> None:
        """
        Displays the graph of the given app.

        Args:
            app: The app to display the graph for. This should be an instance of the App class or a subclass of it.

        Returns:
            None
        """

        nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

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

    def init_base_llm_chain(self) -> None:
        """
        Initializes the base LLM chain.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        base_llm_prompt: Any = load_prompt(self.confis["prompts"]["base_llm_prompt"])
        self.base_llm_chain: Any = base_llm_prompt | self.llm | StrOutputParser()

    def rerank_docs(self, query: str, docs: List[dict], final_k: int) -> List[dict]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            query (str): The query string.
            docs (List[dict]): A list of dictionaries, where each dictionary represents a document.
            final_k (int): The number of top-ranked documents to return.

        Returns:
            List[dict]: A list of the top-ranked documents.
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
               .logits.view(-1)
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

    def llm_generation(self, state: Dict[str, str]) -> Dict[str, str]:
        """
        Generates a response based on internal knowledge.

        Args:
            state (Dict[str, str]): A dictionary containing the question to be answered.

        Returns:
            Dict[str, str]: A dictionary containing the question and the generated response.
        """
        
        print("---GENERATING FROM INTERNAL KNOWLEDGE---")
        question: str = state["question"]
        
        # RAG generation
        generation: str = self.base_llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}