import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import base64
import glob
import ssl
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import nltk
import streamlit as st
import yaml
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_sambanova import ChatSambaNova, SambaNovaEmbeddings
from unstructured.partition.pdf import partition_pdf

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context  # type: ignore[assignment]

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'))

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
)

# Create a logger object
logger = logging.getLogger(__name__)


def load_chat_prompt(path: str) -> ChatPromptTemplate:
    """Load chat prompt from yaml file"""

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    config.pop('_type')

    template = config.pop('template')

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    messages = []
    if isinstance(template, str):
        messages.append(('human', template))

    elif isinstance(template, list):
        for item in template:
            messages.append((item['role'], item['content']))

    return ChatPromptTemplate(messages=messages, **config)


class MultimodalRetrieval:
    """
    Class used to perform multimodal retrieval tasks.
    """

    def __init__(self, conversational: bool = False) -> None:
        """
        initialize MultimodalRetrieval object.
        """
        config_info = self.get_config_info()
        self.llm_info = config_info[0]
        self.lvlm_info = config_info[1]
        self.embedding_model_info = config_info[2]
        self.retrieval_info = config_info[3]
        self.prod_mode = config_info[4]
        self.set_llm()
        self.set_lvlm()
        self.collection_id = str(uuid.uuid4())
        self.vector_collections: Set[Any] = set()
        self.retriever: Optional[MultiVectorRetriever] = None
        self.conversational = conversational
        self.memory: Optional[ConversationSummaryMemory] = None
        self.qa_chain: Optional[Callable[[Any], Dict[str, Any]]] = None
        # set variables for image padding
        os.environ['EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD'] = '150'
        os.environ['EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD'] = '150'

    def get_config_info(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']
        lvlm_info = config['lvlm']
        embedding_model_info = config['embedding_model']
        retrieval_info = config['retrieval']
        prod_mode = config['prod_mode']

        return llm_info, lvlm_info, embedding_model_info, retrieval_info, prod_mode

    def set_llm(self, model: Optional[str] = None) -> None:
        """
        Sets the sambanova LLM

        Parameters:
        Model (str): The name of the model to use for the LLM (overwrites the param set in config).
        """

        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        if model is None:
            model = self.llm_info['model']
        llm_info = {k: v for k, v in self.llm_info.items() if k != 'model'}
        llm = ChatSambaNova(
            api_key=sambanova_api_key,
            **llm_info,
            model=model,
        )
        self.llm = llm

    def set_lvlm(self, model: Optional[str] = None) -> None:
        """
        Sets the LVLM based on the config attributes.

        Parameters:
        model (str): The name of the model to use for the LVLM (overwrites the param set in config).
        """
        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        if model is None:
            model = self.lvlm_info['model']
        lvlm_info = {k: v for k, v in self.lvlm_info.items() if k != 'model'}
        lvlm = ChatSambaNova(
            api_key=sambanova_api_key,
            **lvlm_info,
            model=model,
        )
        self.lvlm = lvlm

    def image_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 encoded string.

        :param: str image_path: The path to the image file.
        :return: The base64 encoded string representation of the image.
        rtype: str
        """
        with open(image_path, 'rb') as image_file:
            image_binary = image_file.read()
            base64_image = base64.b64encode(image_binary).decode()
            return f'data:image/jpeg;base64,{base64_image}'

    def extract_pdf(self, file_path: str) -> Tuple[List[Any], str]:
        # Path to save images
        output_path = os.path.splitext(file_path)[0]
        # Get elements
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            strategy='hi_res',
            hi_res_model_name='yolox',
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            infer_table_structure=True,
            # Titles are any sub-section of the document
            chunking_strategy='by_title',
            max_characters=self.retrieval_info['max_characters'],
            new_after_n_chars=self.retrieval_info['new_after_n_chars'],
            combine_text_under_n_chars=self.retrieval_info['combine_text_under_n_chars'],
            extract_image_block_output_dir=output_path,
        )

        return raw_pdf_elements, output_path

    def init_memory(self) -> None:
        """
        Initialize conversation summary memory for the conversation
        """
        summary_prompt = load_chat_prompt(os.path.join(kit_dir, 'prompts/conversation-summary.yaml'))

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            buffer='The human and AI greet each other to start a conversation.',
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            prompt=summary_prompt,
        )

    def reformulate_query_with_history(self, query: str) -> str:
        """
        Reformulates the query based on the conversation history.

        Args:
        query (str): The current query to reformulate.

        Returns:
        str: The reformulated query.
        """
        if self.memory is None:
            self.init_memory()
        custom_condensed_question_prompt = load_chat_prompt(
            os.path.join(kit_dir, 'prompts', 'multiturn-custom_condensed_query.yaml')
        )
        assert self.memory is not None
        history = self.memory.load_memory_variables({})
        logger.info(f'HISTORY: {history}')
        reformulated_query = self.llm.invoke(
            custom_condensed_question_prompt.format(chat_history=history, question=query)
        ).content
        return str(reformulated_query)

    def summarize_images(self, image_paths: List[str]) -> List[str]:
        """
        Summarizes images by calling the LLM API with a specific prompt.

        Parameters:
        image_paths (list): A list of paths of images to be summarized.

        Returns:
        image_summaries (list[str]): A list of summaries of the input images
        """
        instruction = 'Describe this image in detail. Be specific about graphs include name of axis,\
            labels, legends and important numerical information'

        image_summaries = []
        for image_path in image_paths:
            result = self.lvlm.invoke(
                [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': self.image_to_base64(image_path)}},
                            {'type': 'text', 'text': instruction},
                        ],
                    }
                ]
            )
            summary = result.content
            image_summaries.append(str(summary))
        return image_summaries

    def summarize_texts(self, text_docs: List[Document]) -> Any:
        """
        Summarizes text documents by calling the LLM wit summarize text prompt.

        Parameters:
        text_docs (list): A list of Document objects representing text documents.

        Returns:
        text_summaries (list[str]): A list of summaries of the input text documents.
        """
        text_prompt_template = load_chat_prompt(os.path.join(kit_dir, 'prompts', 'text_summary.yaml'))
        text_summarize_chain: Any = {'element': lambda x: x} | text_prompt_template | self.llm | StrOutputParser()
        texts = [i.page_content for i in text_docs if i.page_content != '']
        if texts:
            text_summaries = text_summarize_chain.batch(texts, {'max_concurrency': 1})
        return text_summaries

    def summarize_tables(self, table_docs: List[Document]) -> Any:
        """
        Summarizes table documents by calling the LLM wit summarize text prompt.

        Parameters:
        table_docs (list): A list of Document objects representing table documents.

        Returns:
        table_summaries (list[str]): A list of summaries of the input table documents.
        """
        table_prompt_template = load_chat_prompt(os.path.join(kit_dir, 'prompts', 'table_summary.yaml'))
        table_summarize_chain: Any = {'element': lambda x: x} | table_prompt_template | self.llm | StrOutputParser()
        tables = [i.page_content for i in table_docs]
        if tables:
            table_summaries = table_summarize_chain.batch(tables, {'max_concurrency': 1})
        return table_summaries

    def process_raw_elements(
        self, raw_elements: List[str], images_paths: Union[List[str], str]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        This function categorizes raw elements (text, tables) convert them to
        a langchain documents, and create a list of each image path contained
        in a set of parent folders  returns separate lists for each type.

        Parameters:
        raw_elements (list): A list of raw unstructured elements extracted from the PDF file.
        images_paths (str or list): A string or list of paths to directories containing images.

        Returns:
        text_docs (list): A list of Document objects representing text documents.
        table_docs (list): A list of Document objects representing table documents.
        image_paths (list): A list of paths to images contained in parent directories.
        """
        categorized_elements = []
        for element in raw_elements:
            if 'unstructured.documents.elements.Table' in str(type(element)):
                assert hasattr(element, 'metadata')
                meta = element.metadata.to_dict()
                meta['type'] = 'table'
                categorized_elements.append(Document(page_content=element.metadata.text_as_html, metadata=meta))
            elif 'unstructured.documents.elements.CompositeElement' in str(type(element)):
                assert hasattr(element, 'metadata')
                meta = element.metadata.to_dict()
                meta['type'] = 'text'
                categorized_elements.append(Document(page_content=str(element), metadata=meta))

        table_docs = [e for e in categorized_elements if e.metadata['type'] == 'table']
        text_docs = [e for e in categorized_elements if e.metadata['type'] == 'text']

        image_paths = []
        if isinstance(images_paths, str):
            images_paths = [images_paths]
        for images_path in images_paths:
            image_paths.extend(glob.glob(os.path.join(images_path, '*.jpg')))
            image_paths.extend(glob.glob(os.path.join(images_path, '*.jpeg')))
            image_paths.extend(glob.glob(os.path.join(images_path, '*.png')))

        return text_docs, table_docs, image_paths

    def create_vectorstore(self) -> MultiVectorRetriever:
        """
        Creates a vectorstore using the in config specified embedding model.

        Returns:
        retriever (MultiVectorRetriever): The retriever object with the vectorstore and docstore.
        """
        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        self.embeddings = SambaNovaEmbeddings(api_key=sambanova_api_key, **self.embedding_model_info)

        collection_name = f'collection_{self.collection_id}'
        logger.info(f'This is the collection name: {collection_name}')

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )
        self.vector_collections.add(collection_name)
        store = InMemoryByteStore()
        id_key = 'doc_id'

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,  # type: ignore
            id_key=id_key,
            search_kwargs={'k': self.retrieval_info['k_retrieved_documents']},
        )

        return retriever

    def vectorstore_ingest(
        self,
        retriever: MultiVectorRetriever,
        text_docs: List[Document],
        table_docs: List[Document],
        image_paths: List[str],
        summarize_texts: bool = False,
        summarize_tables: bool = False,
    ) -> MultiVectorRetriever:
        """
        Ingests documents into the vectorstore and docstore.

        Parameters:
        retriever (MultiVectorRetriever): The retriever object with the vectorstore and docstore.
        text_docs (list): A list of Document objects representing text documents.
        table_docs (list): A list of Document objects representing table documents.
        image_paths (list): A list of paths of images to ingest.
        summarize_texts (bool): A flag indicating whether to summarize text documents.
        summarize_tables (bool): A flag indicating whether to summarize table documents.

        Returns:
        retriever (MultiVectorRetriever): The updated retriever object with the ingested documents.
        """
        id_key = 'doc_id'
        if text_docs:
            doc_ids = [str(uuid.uuid4()) for _ in text_docs]
            if summarize_texts:
                text_summaries = self.summarize_texts(text_docs)
                summary_texts = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)
                ]
                retriever.vectorstore.add_documents(summary_texts)
            else:
                texts = [i.page_content for i in text_docs]
                docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(texts)]
                retriever.vectorstore.add_documents(docs)
            retriever.docstore.mset(list(zip(doc_ids, text_docs)))

        if table_docs:
            table_ids = [str(uuid.uuid4()) for _ in table_docs]
            if summarize_tables:
                table_summaries = self.summarize_tables(table_docs)
                summary_tables = [
                    Document(page_content=s, metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)
                ]
                retriever.vectorstore.add_documents(summary_tables)
            else:
                tables = [i.page_content for i in table_docs]
                docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(tables)]
                retriever.vectorstore.add_documents(docs)
            retriever.docstore.mset(list(zip(table_ids, table_docs)))

        if image_paths:
            img_ids = [str(uuid.uuid4()) for _ in image_paths]
            image_summaries = self.summarize_images(image_paths)
            image_docs = [
                Document(
                    page_content=summary,
                    metadata={
                        'type': 'image',
                        'file_directory': os.path.dirname(image_path),
                        'filename': os.path.basename(image_path),
                    },
                )
                for summary, image_path in zip(image_summaries, image_paths)
            ]
            summary_img = [
                Document(page_content=s, metadata={id_key: img_ids[i], 'path': image_paths[i]})
                for i, s in enumerate(image_summaries)
            ]
            retriever.vectorstore.add_documents(summary_img)
            retriever.docstore.mset(list(zip(img_ids, image_docs)))

        return retriever

    def get_retrieved_images_and_docs(self, retriever: MultiVectorRetriever, query: str) -> Tuple[List[Any], List[Any]]:
        """
        Retrieves image and non-image documents from the vectorstore based on the query.

        Parameters:
        retriever (MultiVectorRetriever): The retriever object with the vectorstore and docstore.
        query (str): The query string to search for relevant documents.

        Returns:
        image_results (list): A list of Document objects representing image documents retrieved from the vectorstore.
        doc_results (list): A list of Document objects representing non-image documents retrieved from the vectorstore.
        """
        results = retriever.invoke(query)
        image_results = [result for result in results if result.metadata['type'] == 'image']
        doc_results = [result for result in results if result.metadata['type'] != 'image']
        return image_results, doc_results

    def get_image_answers(self, retrieved_image_docs: List[Any], query: str) -> List[Any]:
        """
        This function uses LVLM to answer questions based in retrieved images.

        Parameters:
        retrieved_image_docs (list): A list of Document objects representing image documents retrieved from the
        vectorstore.
        query (str): The question string to ask about the images.

        Returns:
        answers (list): A list of answers to the input query for each image.
        """
        image_answer_prompt_template = load_prompt(os.path.join(kit_dir, 'prompts', 'multimodal-qa.yaml'))
        image_answer_prompt = image_answer_prompt_template.format(question=query)
        answers = []
        for doc in retrieved_image_docs:
            image_path = os.path.join(doc.metadata['file_directory'], doc.metadata['filename'])
            answers.append(
                self.lvlm.invoke(
                    [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'image_url', 'image_url': {'url': self.image_to_base64(image_path)}},
                                {'type': 'text', 'text': image_answer_prompt},
                            ],
                        }
                    ]
                ).content
            )
        logger.info(f'PARTIAL ANSWERS FROM IMAGES: {answers}')
        return answers

    def set_retrieval_chain(
        self, retriever: Optional[MultiVectorRetriever] = None, image_retrieval_type: str = 'raw'
    ) -> None:
        """
        This function sets a retrieval_qa_raw_chain function that retrieves answers based on raw image retrieval or a
        retrieval_qa_summary_chain function that retrieves answers based on summary image retrieval.a retrieval chain.

        Parameters:
        retriever (MultiVectorRetriever): The retriever object with the vectorstore and docstore.
        image_retrieval_type (str): The type of image retrieval. It can be either "raw" or "summary".

        """

        prompt = load_chat_prompt(os.path.join(kit_dir, 'prompts', 'knowledge_retriever_custom_qa_prompt.yaml'))
        if retriever is None:
            retriever = self.retriever

        if image_retrieval_type == 'summary':
            retrieval_qa_summary_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                input_key='question',
                output_key='answer',
            )
            retrieval_qa_summary_chain.combine_documents_chain.llm_chain.prompt = prompt
            self.qa_chain = retrieval_qa_summary_chain.invoke

        elif image_retrieval_type == 'raw':

            def retrieval_qa_raw_chain(query: str) -> Dict[str, Any]:
                assert retriever is not None
                image_docs, context_docs = self.get_retrieved_images_and_docs(retriever, query)
                image_answers = self.get_image_answers(image_docs, query)
                text_contexts = [doc.page_content for doc in context_docs]
                full_context = '\n\n'.join(image_answers) + '\n\n' + '\n\n'.join(text_contexts)
                formatted_prompt = prompt.format(context=full_context, question=query)
                answer = self.llm.invoke(formatted_prompt).content
                result = {'question': query, 'answer': answer, 'source_documents': image_docs + context_docs}
                return result

            self.qa_chain = retrieval_qa_raw_chain

        else:
            raise ValueError('Invalid value for image_retrieval_type: {}'.format(image_retrieval_type))

    def call(self, query: str) -> Dict[str, Any]:
        """
        Calls the retrieval chain with the provided query.
        """
        assert self.qa_chain is not None
        logger.info(f'USER QUERY: {query}')
        if self.conversational:
            reformulated_query = self.reformulate_query_with_history(query)
            assert self.memory is not None
            logger.info(f'REFORMULATED QUERY: {reformulated_query}')
            generation = self.qa_chain(reformulated_query)
            self.memory.save_context(inputs={'input': query}, outputs={'answer': generation['answer']})
            logger.info(f'FINAL ANSWER: {generation["answer"]}')
        else:
            generation = self.qa_chain(query)
        return generation

    def st_ingest(
        self,
        files: List[Any],
        summarize_tables: bool = False,
        summarize_texts: bool = False,
        raw_image_retrieval: bool = True,
        data_sub_folder: Optional[str] = None,
    ) -> Any:
        """
        Ingests PDF files, images, and processes them to create a vectorstore and docstore.
        Optionally, it can summarize text and table documents and retrieve answers based on raw or summarized images.

        Parameters:
        files (list): A list of streamlit File objects representing PDF files and images.
        summarize_tables (bool): A flag indicating whether to summarize table documents.
        summarize_texts (bool): A flag indicating whether to summarize text documents.
        raw_image_retrieval (bool): A flag indicating whether to retrieve answers based on raw images or in image
        summaries.
        data_sub_folder (str): A string representing the subfolder where the data will be stored.
        """
        pdf_files = [file for file in files if file.name.endswith(('.pdf'))]
        image_files = [file for file in files if file.name.endswith(('.jpg', '.jpeg', 'png'))]
        raw_elements = []
        image_paths = []
        if data_sub_folder is None:
            data_sub_folder = 'upload'
        upload_folder = os.path.join(kit_dir, 'data', data_sub_folder)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        print('Extracting content from documents')
        for pdf in pdf_files:
            file_path = os.path.join(upload_folder, pdf.name)
            with open(file_path, 'wb') as file:
                file.write(pdf.read())
            raw_pdf_elements, output_path = self.extract_pdf(file_path)
            raw_elements.extend(raw_pdf_elements)
            image_paths.append(output_path)
        if image_files:
            single_images_folder = os.path.join(upload_folder, f'images_{time.time()}')
            os.makedirs(single_images_folder)
            for image in image_files:
                file_path = os.path.join(single_images_folder, image.name)
                with open(file_path, 'wb') as file:
                    file.write(image.read())
            image_paths.append(single_images_folder)
        text_docs, table_docs, image_paths = self.process_raw_elements(raw_elements, image_paths)
        if len(image_paths) > 0:
            print(
                f'* {len(image_paths)} calls to the multimodal model will be done to summarize and '
                'ingest images in provided documents\n\n'
            )
        if summarize_texts and len(text_docs) > 0:
            print(
                f'* {len(text_docs)} calls to the LLM will be done to summarize and '
                'ingest texts in provided documents\n\n'
            )
        if summarize_tables and len(table_docs) > 0:
            print(
                f'* {len(table_docs)} calls to the LLM will be done to summarize and '
                'ingest tables in provided documents\n\n'
            )
        print(
            f'* **In total {len(image_paths) + len(text_docs) + len(table_docs)} '
            'chunks will be sent to the embeddings model to ingest**\n'
        )
        self.retriever = self.create_vectorstore()
        self.retriever = self.vectorstore_ingest(
            self.retriever,
            text_docs,
            table_docs,
            image_paths,
            summarize_texts=summarize_texts,
            summarize_tables=summarize_tables,
        )
        if raw_image_retrieval:
            self.set_retrieval_chain(retriever=self.retriever, image_retrieval_type='raw')
        else:
            self.set_retrieval_chain(retriever=self.retriever, image_retrieval_type='summary')
