import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import nltk
import torch
import yaml
from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelLike
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb
from utils.visual.env_utils import get_wandb_key

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'))


from utils.parsing.sambaparse import parse_doc_universal

# Handle the WANDB_API_KEY resolution before importing weave
wandb_api_key = get_wandb_key()

# If WANDB_API_KEY is set, proceed with weave initialization
if wandb_api_key:
    import weave

    # Initialize Weave with your project name
    weave.init('sambanova_ekr')
else:
    print('WANDB_API_KEY is not set. Weave initialization skipped.')

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


class RetrievalQAChain(Chain):
    """class for question-answering."""

    retriever: BaseRetriever
    rerank: bool = True
    llm: LanguageModelLike
    qa_prompt: ChatPromptTemplate
    final_k_retrieved_documents: int = 3

    @property
    def input_keys(self) -> List[str]:
        """Input keys.
        :meta private:
        """
        return ['question']

    @property
    def output_keys(self) -> List[str]:
        """Output keys.
        :meta private:
        """
        return ['answer', 'source_documents']

    def _format_docs(self, docs: List[Document]) -> str:
        return '\n\n'.join(doc.page_content for doc in docs)

    def rerank_docs(self, query: str, docs: List[Document], final_k: int) -> List[Document]:
        # Lazy hardcoding for now
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
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

        docs_sorted = [docs[k] for k in scores_sorted_idx]
        # docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
        docs_sorted = docs_sorted[:final_k]

        return docs_sorted

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        response: Dict[str, Any] = {}
        documents = self.retriever.invoke(inputs['question'])
        if self.rerank:
            documents = self.rerank_docs(inputs['question'], documents, self.final_k_retrieved_documents)
        docs = self._format_docs(documents)
        response['answer'] = qa_chain.invoke({'question': inputs['question'], 'context': docs})
        response['source_documents'] = documents
        return response


class DocumentRetrieval:
    def __init__(self, sambanova_api_key: str) -> None:
        self.vectordb = VectorDb()
        config_info = self.get_config_info()
        self.llm_info = config_info[0]
        self.embedding_model_info = config_info[1]
        self.retrieval_info = config_info[2]
        self.prompts = config_info[3]
        self.prod_mode = config_info[4]
        self.pdf_only_mode = config_info[5]
        self.retriever = None
        self.sambanova_api_key = sambanova_api_key
        self.llm = self.set_llm()

    def get_config_info(self) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str], bool, bool]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']
        embedding_model_info = config['embedding_model']
        retrieval_info = config['retrieval']
        prompts = config['prompts']
        prod_mode = config['prod_mode']
        pdf_only_mode = config['pdf_only_mode']

        return llm_info, embedding_model_info, retrieval_info, prompts, prod_mode, pdf_only_mode

    def set_llm(self) -> LLM:
        llm = APIGateway.load_chat(
            type=self.llm_info['api'],
            do_sample=self.llm_info['do_sample'],
            max_tokens=self.llm_info['max_tokens'],
            temperature=self.llm_info['temperature'],
            model=self.llm_info['model'],
            process_prompt=False,
            sambanova_api_key=self.sambanova_api_key,
        )
        return llm

    def parse_doc(self, doc_folder: str, additional_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Parse specified documents and return a list of LangChain documents.

        Args:
            doc_folder (str): Path to the documents.
            additional_metadata (Optional[Dict], optional): Additional metadata to include in the processed documents.
                Defaults to an empty dictionary.

        Returns:
            List[Document]: A list of LangChain documents.
        """
        if additional_metadata is None:
            additional_metadata = {}

        _, _, langchain_docs = parse_doc_universal(
            doc=doc_folder, additional_metadata=additional_metadata, lite_mode=self.pdf_only_mode
        )

        return langchain_docs

    def load_embedding_model(self) -> Embeddings:
        embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )
        return embeddings
    
    def load_chat_prompt(self, path: str) -> ChatPromptTemplate:
        """Load chat prompt from yaml file"""
        
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        config.pop("_type")

        template = config.pop("template")

        if not template:
            msg = "Can't load chat prompt without template"
            raise ValueError(msg)

        messages = []
        if isinstance(template, str):
            messages.append(("human", template))

        elif isinstance(template, list):
            for item in template:
                messages.append((item["role"], item["content"]))

        return ChatPromptTemplate(messages=messages, **config)

    def create_vector_store(
        self,
        text_chunks: List[Document],
        embeddings: Any,
        output_db: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Any:
        print(f'Collection name is {collection_name}')
        vectorstore = self.vectordb.create_vector_store(
            text_chunks, embeddings, output_db=output_db, collection_name=collection_name, db_type='chroma'
        )
        return vectorstore

    def load_vdb(self, db_path: str, embeddings: Any, collection_name: Optional[str] = None) -> Any:
        print(f'Loading collection name is {collection_name}')
        vectorstore = self.vectordb.load_vdb(db_path, embeddings, db_type='chroma', collection_name=collection_name)
        return vectorstore

    def init_retriever(self, vectorstore: Any) -> None:
        if self.retrieval_info['rerank']:
            self.retriever = vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={
                    'score_threshold': self.retrieval_info['score_threshold'],
                    'k': self.retrieval_info['k_retrieved_documents'],
                },
            )
        else:
            self.retriever = vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={
                    'score_threshold': self.retrieval_info['score_threshold'],
                    'k': self.retrieval_info['final_k_retrieved_documents'],
                },
            )

    def get_qa_retrieval_chain(self) -> RetrievalQAChain:
        """
        Generate a qa_retrieval chain using a language model.

        This function uses a language model, specifically a SambaNova LLM, to generate a qa_retrieval chain
        based on the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                            for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA without memory
        """
        assert isinstance(
            self.retriever, VectorStoreRetriever
        ), f'The Retriever must be VectorStoreRetriever. Got type {type(self.retriever)}'
        retrievalQAChain = RetrievalQAChain(
            retriever=self.retriever,
            llm=self.llm,
            qa_prompt=self.load_chat_prompt(os.path.join(repo_dir, self.prompts['qa_prompt'])),
            rerank=self.retrieval_info['rerank'],
            final_k_retrieved_documents=self.retrieval_info['final_k_retrieved_documents'],
        )
        return retrievalQAChain

    def get_conversational_qa_retrieval_chain(self) -> None:
        """
        Generate a conversational retrieval qa chain using a language model.

        This function uses a language model, specifically a SambaNova LLM, to generate a conversational_qa_retrieval
        chain based on the chat history and the relevant retrieved content from the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                                        for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA with memory
        """
