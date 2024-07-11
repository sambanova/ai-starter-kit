import os
import yaml
import torch
import base64
import nest_asyncio
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
        try:
            with open(filename, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The YAML configuration file {filename} was not found.") 
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")

    def load_embedding_model(self):
        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info) 
        return embeddings  

    def init_llm(self):
        
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

    def _format_docs(self, docs):
    
        return "\n\n".join(doc.page_content for doc in docs)

    def init_embeddings(self):

        self.embeddings = SambaStudioEmbeddings()
    
    def _display_image(self, image_bytes: bytes, width=256):
        decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
        html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
        display(HTML(html))
    
    def display_graph(self, app):

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
        base_llm_prompt = load_prompt(self.confis["prompts"]["base_llm_prompt"])
        self.base_llm_chain = base_llm_prompt| self.llm | StrOutputParser()

    def rerank_docs(self, query, docs, final_k):
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

    def llm_generation(self, state):
        
        print("---GENERATING FROM INTERNAL KNOWLEDGE---")
        question = state["question"]
        
        # RAG generation
        generation = self.base_llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}