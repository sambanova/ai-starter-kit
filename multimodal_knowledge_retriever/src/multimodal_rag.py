import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import glob
import ssl
import time
import uuid
from typing import Callable, Dict, List, Tuple, Union

import nltk
import yaml
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from unstructured.partition.pdf import partition_pdf

from utils.model_wrappers.api_gateway import APIGateway
from utils.model_wrappers.multimodal_models import SambastudioMultimodal

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'))


class MultimodalRetrieval:
    """
    Class used to perform multimodal retrieval tasks.
    """

    def __init__(self) -> None:
        """
        initialize MultimodalRetrieval object.
        """
        config_info = self.get_config_info()
        self.llm_info = config_info[0]
        self.lvlm_info = config_info[1]
        self.embedding_model_info = config_info[2]
        self.retrieval_info = config_info[3]
        self.llm = self.set_llm()
        self.lvlm = SambastudioMultimodal(
            model=self.lvlm_info['model'],
            temperature=self.lvlm_info['temperature'],
            max_tokens_to_generate=self.lvlm_info['max_tokens_to_generate'],
            top_p=self.lvlm_info['top_p'],
            top_k=self.lvlm_info['top_k'],
            do_sample=self.lvlm_info['do_sample'],
        )

    def get_config_info(self) -> Tuple[str, str, str, str]:
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

        return llm_info, lvlm_info, embedding_model_info, retrieval_info

    def set_llm(self) -> LLM:
        """
        Sets the sncloud, or sambastudio LLM based on the llm type attribute.

        Returns:
        LLM: The SambaStudio Cloud or Sambastudio Langchain LLM.
        """
        llm = APIGateway.load_llm(
            type=self.llm_info['type'],
            streaming=True,
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
        )
        return llm

    def extract_pdf(self, file_path: str) -> Tuple[List, str]:
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

    def summarize_images(self, image_paths: List[str]) -> List[str]:
        """
        Summarizes images by calling the LLM API with a specific prompt.

        Parameters:
        image_paths (list): A list of paths of images to be summarized.

        Returns:
        image_summaries (list[str]): A list of summaries of the input images
        """
        instruction = 'Describe the image in detail. Be specific about graphs include name of axis,\
            labels, legends and important numerical information'

        image_summaries = []
        for image_path in image_paths:
            summary = self.lvlm.invoke(instruction, image_path)
            image_summaries.append(summary)
        return image_summaries

    def summarize_texts(self, text_docs: List) -> List[str]:
        """
        Summarizes text documents by calling the LLM wit summarize text prompt.

        Parameters:
        text_docs (list): A list of Document objects representing text documents.

        Returns:
        text_summaries (list[str]): A list of summaries of the input text documents.
        """
        text_prompt_template = load_prompt(os.path.join(kit_dir, 'prompts', 'llama70b-text_summary.yaml'))
        text_summarize_chain = {'element': lambda x: x} | text_prompt_template | self.llm | StrOutputParser()
        texts = [i.page_content for i in text_docs if i.page_content != '']
        if texts:
            text_summaries = text_summarize_chain.batch(texts, {'max_concurrency': 1})
        return text_summaries

    def summarize_tables(self, table_docs: List) -> List[str]:
        """
        Summarizes table documents by calling the LLM wit summarize text prompt.

        Parameters:
        table_docs (list): A list of Document objects representing table documents.

        Returns:
        table_summaries (list[str]): A list of summaries of the input table documents.
        """
        table_prompt_template = load_prompt(os.path.join(kit_dir, 'prompts', 'llama70b-table_summary.yaml'))
        table_summarize_chain = {'element': lambda x: x} | table_prompt_template | self.llm | StrOutputParser()
        tables = [i.page_content for i in table_docs]
        if tables:
            table_summaries = table_summarize_chain.batch(tables, {'max_concurrency': 1})
        return table_summaries

    def process_raw_elements(self, raw_elements: List, images_paths: Union[List, str]) -> Tuple[List, List, List]:
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
                meta = element.metadata.to_dict()
                meta['type'] = 'table'
                categorized_elements.append(Document(page_content=element.metadata.text_as_html, metadata=meta))
            elif 'unstructured.documents.elements.CompositeElement' in str(type(element)):
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
        self.embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )

        vectorstore = Chroma(
            collection_name='summaries',
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )
        store = InMemoryByteStore()
        id_key = 'doc_id'

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
            search_kwargs={'k': self.retrieval_info['k_retrieved_documents']},
        )

        return retriever

    def vectorstore_ingest(
        self,
        retriever: MultiVectorRetriever,
        text_docs: List,
        table_docs: List,
        image_paths: List,
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
                texts = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(texts)]
                retriever.vectorstore.add_documents(texts)
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
                tables = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(tables)]
                retriever.vectorstore.add_documents(tables)
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

    def get_retrieved_images_and_docs(self, retriever: MultiVectorRetriever, query: str) -> Tuple[List, List]:
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

    def get_image_answers(self, retrieved_image_docs: List, query: str) -> List:
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
            answers.append(self.lvlm.invoke(image_answer_prompt, image_path))
        return answers

    def get_retrieval_chain(
        self, retriever: MultiVectorRetriever, image_retrieval_type: str = 'raw'
    ) -> Tuple[Callable, Callable]:
        """
        This function returns a retrieval chain.

        Parameters:
        retriever (MultiVectorRetriever): The retriever object with the vectorstore and docstore.
        image_retrieval_type (str): The type of image retrieval. It can be either "raw" or "summary".

        Returns:
        retrieval_qa_raw_chain (function): A function that retrieves answers based on raw image retrieval.
        retrieval_qa_summary_chain (function): A function that retrieves answers based on summary image retrieval.
        """
        prompt = load_prompt(os.path.join(kit_dir, 'prompts', 'llama70b-knowledge_retriever_custom_qa_prompt.yaml'))
        if image_retrieval_type == 'summary':
            retrieval_qa_summary_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                input_key='question',
                output_key='answer',
            )
            retrieval_qa_summary_chain.combine_documents_chain.llm_chain.prompt = prompt
            return retrieval_qa_summary_chain.invoke

        if image_retrieval_type == 'raw':

            def retrieval_qa_raw_chain(query: str) -> Dict:
                image_docs, context_docs = self.get_retrieved_images_and_docs(retriever, query)
                image_answers = self.get_image_answers(image_docs, query)
                text_contexts = [doc.page_content for doc in context_docs]
                full_context = '\n\n'.join(image_answers) + '\n\n' + '\n\n'.join(text_contexts)
                formated_prompt = prompt.format(context=full_context, question=query)
                answer = self.llm.invoke(formated_prompt)
                result = {'question': query, 'answer': answer, 'source_documents': image_docs + context_docs}
                return result

            return retrieval_qa_raw_chain

        else:
            raise ValueError('Invalid value for image_retrieval_type: {}'.format(image_retrieval_type))

    def st_ingest(
        self,
        files: List,
        summarize_tables: bool = False,
        summarize_texts: bool = False,
        raw_image_retrieval: bool = True,
    ) -> Callable:
        """
        Ingests PDF files, images, and processes them to create a vectorstore and docstore.
        Optionally, it can summarize text and table documents and retrieve answers based on raw or summarized images.

        Parameters:
        files (list): A list of streamlit File objects representing PDF files and images.
        summarize_tables (bool): A flag indicating whether to summarize table documents.
        summarize_texts (bool): A flag indicating whether to summarize text documents.
        raw_image_retrieval (bool): A flag indicating whether to retrieve answers based on raw images or in image
        summaries.

        Returns:
        qa_chain (function): A function that retrieves answers based on the specified retrieval type.
        """
        pdf_files = [file for file in files if file.name.endswith(('.pdf'))]
        image_files = [file for file in files if file.name.endswith(('.jpg', '.jpeg', 'png'))]
        raw_elements = []
        image_paths = []
        upload_folder = os.path.join(kit_dir, 'data', 'upload')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
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
        retriever = self.create_vectorstore()
        retriever = self.vectorstore_ingest(
            retriever,
            text_docs,
            table_docs,
            image_paths,
            summarize_texts=summarize_texts,
            summarize_tables=summarize_tables,
        )
        if raw_image_retrieval:
            qa_chain = self.get_retrieval_chain(retriever, image_retrieval_type='raw')
        else:
            qa_chain = self.get_retrieval_chain(retriever, image_retrieval_type='summary')
        return qa_chain
