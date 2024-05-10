import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import glob
import yaml
import json
import uuid
import time
import base64
import requests
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from chromadb.config import Settings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.llms.sambanova import Sambaverse, SambaStudio
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from utils.sambanova_endpoint import SambaNovaEmbeddingModel

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))

class MultimodalRetrieval():
    def __init__(self):
        config_info = self.get_config_info()
        self.api_info =config_info[0]
        self.llm_info =config_info[1]
        self.lvlm_info =config_info[2]
        self.embedding_model_info =config_info[3] 
        self.retrieval_info =config_info[4]
        self.loader = config_info[5]
        self.llm = self.set_llm()
        
    def get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        lvlm_info = config["lvlm"]
        embedding_model_info = config["embedding_model"]
        retrieval_info = config["retrieval"]
        loader = config["loaders"]
        
        return api_info, llm_info, lvlm_info, embedding_model_info, retrieval_info, loader
    
    @staticmethod
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            image_binary = image_file.read()
            base64_image = base64.b64encode(image_binary).decode()
            return base64_image
    
    def llava_call(self, prompt, image_path, base_url=None, project_id=None, endpoint_id=None, api_key=None):
        if base_url is None:
            base_url = os.environ.get('LVLM_BASE_URL')
        if project_id is None:
            project_id = os.environ.get('LVLM_PROJECT_ID')
        if endpoint_id is None:
            endpoint_id = os.environ.get('LVLM_ENDPOINT_ID')
        if api_key is None:
            api_key = os.environ.get('LVLM_API_KEY')
        endpoint_url = f"{base_url}/api/predict/generic/{project_id}/{endpoint_id}"
        endpoint_key = api_key
        # Define the data payload
        image_b64=MultimodalRetrieval.image_to_base64(image_path)
        data = {
            "instances": [{
                "prompt": prompt,
                "image_content": f"{image_b64}"
            }],
            "params": {
                "do_sample": {"type": "bool", "value": str(self.lvlm_info["do_sample"])},
                "max_tokens_to_generate": {"type": "int", "value": str(self.lvlm_info["max_tokens_to_generate"])},
                "temperature": {"type": "float", "value": str(self.lvlm_info["temperature"])},
                "top_k": {"type": "int", "value":  str(self.lvlm_info["top_k"])},
                "top_logprobs": {"type": "int", "value": "0"},
                "top_p": {"type": "float", "value":  str(self.lvlm_info["top_p"])}
            }
        }
        # Define headers
        headers = {
            "Content-Type": "application/json",
            "key": endpoint_key
        }
        response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))
        return response.json()["predictions"][0]['completion']
    
    def set_llm(self):
        if self.api_info == "sambaverse":
            llm = Sambaverse(
                    sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "temperature": self.llm_info["temperature"],
                        "process_prompt": True,
                        "select_expert": self.llm_info["sambaverse_select_expert"]
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
        elif self.api_info == "sambastudio":
            llm = SambaStudio(
                model_kwargs={
                    "do_sample": False,
                    "temperature": self.llm_info["temperature"],
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
        return llm
    
    def extract_pdf(self, file_path):
    # Path to save images
        output_path=os.path.splitext(file_path)[0]
        # Get elements
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            strategy='hi_res',
            hi_res_model_name="yolox",
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            infer_table_structure=True,
            # Titles are any sub-section of the document
            chunking_strategy="by_title",
            max_characters=self.retrieval_info["max_characters"],
            new_after_n_chars=self.retrieval_info["new_after_n_chars"],
            combine_text_under_n_chars=self.retrieval_info["combine_text_under_n_chars"],
            extract_image_block_output_dir=output_path,
        )
        
        return raw_pdf_elements, output_path
    
    def summarize_images(self, image_paths):
        prompt_template = load_prompt(os.path.join(kit_dir, "prompts", "llava.yaml"))
        instruction="Describe the image in detail. Be specific about graphs include name of axis,\
            labels, legends and important numerical information"
        formated_prompt = prompt_template.format(instruction = instruction)
        
        image_summaries = []
        for image_path in image_paths:
            summary = self.llava_call(formated_prompt, image_path)
            image_summaries.append(summary)
        return image_summaries
        
    def summarize_texts(self, text_docs):
        text_prompt_template = load_prompt(os.path.join(kit_dir, "prompts", "llama70b-text_summary.yaml"))
        text_summarize_chain = {"element": lambda x: x} | text_prompt_template | self.llm | StrOutputParser()
        texts = [i.page_content for i in text_docs if i.page_content != ""]
        if texts:
            text_summaries = text_summarize_chain.batch(texts, {"max_concurrency": 1})
        return text_summaries
    
    def summarize_tables(self, table_docs):
        table_prompt_template = load_prompt(os.path.join(kit_dir, "prompts", "llama70b-table_summary.yaml"))
        table_summarize_chain = {"element": lambda x: x} | table_prompt_template | self.llm | StrOutputParser()
        tables = [i.page_content for i in table_docs]
        if tables:
            table_summaries = table_summarize_chain.batch(tables, {"max_concurrency":1})
        return table_summaries
    
    def process_raw_elements(self, raw_elements, images_paths):
        #Categorize by type
        categorized_elements = []
        for element in raw_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                meta = element.metadata.to_dict()
                meta["type"] = "table"
                categorized_elements.append(Document(page_content=element.metadata.text_as_html, metadata=meta))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                meta = element.metadata.to_dict()
                meta["type"] = "text"
                categorized_elements.append(Document(page_content=str(element), metadata=meta))
                
        table_docs = [e for e in categorized_elements if e.metadata["type"] == "table"]
        text_docs = [e for e in categorized_elements if e.metadata["type"] == "text"]
                
        image_paths = []
        if isinstance(images_paths, str): 
            images_paths = [images_paths]
        for images_path in images_paths:
            image_paths.extend(glob.glob(os.path.join(images_path, '*.jpg')))
            image_paths.extend(glob.glob(os.path.join(images_path, '*.jpeg')))
            image_paths.extend(glob.glob(os.path.join(images_path, '*.png')))
        
        return text_docs, table_docs, image_paths
    
    def create_vectorstore(self):
        if self.embedding_model_info == "sambastudio":
            self.embeddings = SambaNovaEmbeddingModel()
        elif self.embedding_model_info == "cpu":
            encode_kwargs = {"normalize_embeddings": True}
            embedding_model = "hkunlp/instructor-large"
            self.embeddings = HuggingFaceInstructEmbeddings(
                model_name=embedding_model,
                embed_instruction="",  # no instruction is needed for candidate passages
                query_instruction="Represent this sentence for searching relevant passages: ",
                encode_kwargs=encode_kwargs,
            )
        else:
            raise ValueError(f"{self.embedding_model_info} is not a valid embedding model type")
        
        vectorstore = Chroma(
            collection_name="summaries", 
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False)
        )
        store = InMemoryByteStore()  
        id_key = "doc_id"
        
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": self.retrieval_info["k_retrieved_documents"]}
        )
        
        return retriever
    
    def vectorstore_ingest(self, retriever, text_docs, table_docs, image_paths, summarize_texts=False, summarize_tables=False):
        id_key = "doc_id"
        if text_docs:
            doc_ids = [str(uuid.uuid4()) for _ in text_docs]
            if summarize_texts:
                text_summaries = self.summarize_texts(text_docs)
                summary_texts = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(text_summaries)
                ]
                retriever.vectorstore.add_documents(summary_texts)
            else:
                texts = [i.page_content for i in text_docs] 
                texts = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(texts)
                ]
                retriever.vectorstore.add_documents(texts) 
            retriever.docstore.mset(list(zip(doc_ids, text_docs)))

        if table_docs:
            table_ids = [str(uuid.uuid4()) for _ in table_docs]
            if summarize_tables:
                table_summaries = self.summarize_tables(table_docs)
                summary_tables = [
                    Document(page_content=s, metadata={id_key: table_ids[i]})
                    for i, s in enumerate(table_summaries)
                ]
                retriever.vectorstore.add_documents(summary_tables)
            else: 
                tables = [i.page_content for i in table_docs] 
                tables = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(tables)
                ]
                retriever.vectorstore.add_documents(tables)
            retriever.docstore.mset(list(zip(table_ids, table_docs)))
            
        if image_paths:
            img_ids = [str(uuid.uuid4()) for _ in image_paths]
            image_summaries = self.summarize_images(image_paths)
            image_docs=[
                Document(
                    page_content=summary,
                    metadata={
                        "type": "image", 
                        'file_directory': os.path.dirname(image_path),
                        'filename': os.path.basename(image_path)
                        }
                    ) 
                for summary, image_path in zip(image_summaries, image_paths)
            ]
            summary_img = [
                Document(page_content=s, metadata={id_key: img_ids[i], "path" : image_paths[i]})
                for i, s in enumerate(image_summaries)
            ]            
            retriever.vectorstore.add_documents(summary_img)
            retriever.docstore.mset(list(zip(img_ids, image_docs)))
        
        return retriever
    
    def get_retrieved_images_and_docs(self, retriever, query):
        results=retriever.invoke(query)
        image_results = [result for result in results if result.metadata["type"]=="image"]
        doc_results = [result for result in results if result.metadata["type"]!="image"]
        return image_results, doc_results
    
    def get_image_answers(self, retrieved_image_docs, query):
        image_answer_prompt_template = load_prompt(os.path.join(kit_dir,"prompts","llava-qa.yaml"))
        image_answer_prompt = image_answer_prompt_template.format(question = query)
        answers = []
        for doc in retrieved_image_docs:
            image_path = os.path.join(doc.metadata["file_directory"],doc.metadata["filename"])
            answers.append(self.llava_call(image_answer_prompt, image_path))
        return answers
    
    def set_retrieval_chain(self, retriever, image_retrieval_type="raw"):
        prompt = load_prompt(os.path.join(kit_dir,"prompts","llama70b-knowledge_retriever_custom_qa_prompt.yaml"))
        if image_retrieval_type == "summary":
            retrieval_qa_summary_chain = RetrievalQA.from_llm(
                llm = self.llm,
                retriever=retriever,
                return_source_documents=True,
                input_key="question",
                output_key="answer"
            )
            retrieval_qa_summary_chain.combine_documents_chain.llm_chain.prompt=prompt
            return retrieval_qa_summary_chain.invoke
        
        if image_retrieval_type == "raw":
            def retrieval_qa_raw_chain(query):
                image_docs, context_docs = self.get_retrieved_images_and_docs(retriever, query)
                image_answers = self.get_image_answers(image_docs, query)
                text_contexts = [doc.page_content for doc in context_docs]
                full_context = '\n\n'.join(image_answers)+'\n\n'+'\n\n'.join(text_contexts)
                formated_prompt = prompt.format(context=full_context, question=query)
                answer = self.llm.invoke(formated_prompt)
                result={'question': query, 'answer': answer, 'source_documents': image_docs+context_docs}
                return result
            return retrieval_qa_raw_chain
        
        else:
            raise ValueError("Invalid value for image_retrieval_type: {}".format(image_retrieval_type))
    
    
    def st_ingest(self, files, summarize_tables=False, summarize_texts=False, raw_image_retrieval=True):
        pdf_files = [file for file in files if file.name.endswith((".pdf"))] 
        image_files =  [file for file in files if file.name.endswith((".jpg",".jpeg","png"))]
        raw_elements = []
        image_paths = []
        upload_folder = os.path.join(kit_dir, "data", "upload")
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
            single_images_folder = os.path.join(upload_folder, f"images_{time.time()}")
            os.makedirs(single_images_folder)
            for image in image_files:
                file_path = os.path.join(single_images_folder, image.name)
                with open(file_path, 'wb') as file:
                    file.write(image.read())
            image_paths.append(single_images_folder)
        text_docs, table_docs, image_paths = self.process_raw_elements(raw_elements, image_paths)
        retriever=self.create_vectorstore()
        retriever=self.vectorstore_ingest(
            retriever, text_docs, table_docs, image_paths, summarize_texts=summarize_texts, summarize_tables=summarize_tables
            )
        if raw_image_retrieval:
            qa_chain =self.set_retrieval_chain(retriever, image_retrieval_type="raw")
        else:
            qa_chain = self.set_retrieval_chain(retriever, image_retrieval_type="summary")
        return qa_chain