import re
import sys
import os, shutil
sys.path.append("../")
sys.path.append("../../../")

from langchain.prompts import PromptTemplate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from utils.sambanova_endpoint import SambaNovaEndpoint
from .rerank_retriever import RetrievalQAReranker, VectorStoreRetrieverReranker

class RAGPipeline:
    """
    Class for managing a RAG pipeline.
    """

    def __init__(self, 
                 llm: SambaNovaEndpoint, 
                 vector_db_location: str, 
                 embedding_model: HuggingFaceInstructEmbeddings, 
                 k: int = 20):
        """
        Initialize the RAGPipeline class.

        Args:
            llm (SambaNovaEndpoint): The LLM model used for generating answers (e.g., Llama, GPT).
            vector_db_location (str): Location of the vector database used for document retrieval.
            embedding_model (HuggingFaceInstructEmbeddings): The embedding model used for encoding documents.
            k (int): The number of documents to retrieve during document retrieval. Default value is 20.
        """
        self.init_prompt_template()
        self.init_retriever_and_QAchain(llm, 
                                        vector_db_location, 
                                        embedding_model, 
                                        self.prompt_template,
                                        k)
        self.llm = llm
        self.embedding_model = embedding_model
    
    def init_prompt_template(self):
        """
        Initialize the prompt template for Llama 2.
        """
        
        sys_prompt = """
        <s>[INST] <<SYS>> You are a helpful ai assistant that is tasked with answering questions, given some helpful tips and contexts that are chunks from data sheets.  You are most well suited for answering questions using plain text and bulleted lists, but you can also use tabular data in markdown format if you cannot find a clear answer in plan text.  You avoid using formulas and equations because you are not good at advanced maths and don't believe them to be a good source of information.

        Given extracted segments from the Analog Device's datasheets and a question, your task is to create a final answer to the user's question. Do not include the user's question in your answer.

        Here are some more rules:
        - If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
        - Do NOT include links in the body of your answer.
        - Use ONLY the information provided in the context section. 
        - All parts of your response must be relevant to the question, and must be factually correct. You will be penalized if you mention something in your response that is not relevant to the question. 
        - Pay extra attention when relationship operators appear in user queries. "Greater than" or ">" means A being strictly larger than B, "less than" or "<" means A being strictly smaller than B. "Equal to" or "==" is different from "greater than" (>) or "less than" (<).
        - Do NOT make any calculation. Do NOT make any assumption. Do NOT use any math formula. Avoid inferences and estimations. Answer question ONLY based on extracted values if the information is in the datasheet; Otherwise reply that you don't know.
        - Extract all key and/or relevant information to the content of the question and state where this information came from.  If asked about numeric values for voltage, current, frequencies, etc. give prefernce to tablular information with column headings: PARAMETER, MIN, TYPICAL, MAX, and UNITS. 
        - First identify specifications in relevant tabular or text information. Only after you have identified all relevant information related to the specifications will you provide an answer.
        - Only output specific answer within the context of user query.
        - Some tables have similar names, but for difference products. If the product name is in the table name, use as a source only tables whose names contain the same product names as in the user's query.

        Given the following question and segments from the product datasheet, write a helpful answer:<</SYS>>

        QUESTION: {question}
        =========
        {context}
        =========
        FINAL_ANSWER: 

        Think step by step and be patient when answering.
        Only when you are done with all steps, provide the answer based on your intermediate steps.  
        Explain your thought process and actions at each intermediate step.  Do not provide contexts back.
        Before presenting your final answer, make sure that you correctly followed all steps. 
        Avoid repeating the question in your final answer. 
        Avoid providing an explanation of how you arrived in your final answer. 
        Display final answer at the end of your output. [/INST]</s>
        """
        self.prompt_template = PromptTemplate(
            template=sys_prompt, input_variables=["context", "question"]
        )
       
    
    def init_retriever_and_QAchain(self,
                                   llm: SambaNovaEndpoint, 
                                    vector_db_location: str, 
                                    embedding_model: HuggingFaceInstructEmbeddings, 
                                    prompt_template: PromptTemplate,
                                    k: int = 20 ):
        """
        Initialize the retriever and QAchain.

        Args:
            llm (SambaNovaEndpoint): The language model used for generating answers (e.g., GPT, BERT).
            vector_db_location (str): Location of the vector database used for document retrieval.
            embedding_model (HuggingFaceInstructEmbeddings): The embedding model used for encoding documents.
            prompt_template (PromptTemplate): The prompt template used for LLM.
            k (int): The number of documents to retrieve during document retrieval. Default value is 20.
        """
        q3_retriever = VectorStoreRetrieverReranker(
            vectorstore = Chroma(persist_directory=vector_db_location, embedding_function=embedding_model), 
            search_kwargs={"k": k})
        self.qa_chain = RetrievalQAReranker.from_chain_type(
                    llm=llm,
                    retriever=q3_retriever,
                    return_source_documents=True,
                    input_key='question',
                    output_key='answer',
        )
        self.qa_chain.combine_documents_chain.llm_chain.prompt = prompt_template 


    def OutputParser(self, text: str) -> str:
        """
        Parse the output from LLM model.

        Args:
            text (str): The output from LLM model. 

        Returns:
            str: The final answer of LLM model.
        """
        pattern = r'(?i)(FINAL[_ ]ANSWER:|The answer is:)'

        final_answer_match = re.search(pattern, text, flags=re.IGNORECASE)
        if final_answer_match:
            explanation_match = re.search(r'Explanation:', text, flags=re.IGNORECASE)
            if explanation_match:
                final_answer = text[final_answer_match.end():explanation_match.start()].strip()
            else:
                final_answer = text[final_answer_match.end():].strip()
            return final_answer
        else:
            return text

# def create_embedding_db(md_filepath: str, 
#                         vector_db_location: str):
#     """
#     Create an embedding database from the provided markdown file.

#     Args:
#         md_filepath (str): The filepath of the markdown file containing documents.
#         vector_db_location (str): Location to save the embedding database.

#     Returns:
#         vectorstore (Chroma): the embedding database
#     """
#     if os.path.exists(vector_db_location):
#         shutil.rmtree(vector_db_location)
#     vectorstore = create_db(md_filepath, ".pdf.md", vector_db_location , chunk_size=500, chunk_overlap=0, db_type='chromadb', remove_toc=True)
#     return vectorstore

def load_embedding_db(vector_db_location: str, 
                      embedding_model: HuggingFaceInstructEmbeddings):
    """
    Load an embedding database with the specified embedding model.

    Args:
        vector_db_location (str): Location of the embedding database.
        embedding_model (HuggingFaceInstructEmbeddings): The embedding model to use for loading the database.

    Returns:
        vectorstore (Chroma): the embedding database
    """
    vectorstore = Chroma(persist_directory=vector_db_location, embedding_function=embedding_model)
    return vectorstore

def load_reranker_model():
    """
    Load the reranker model BAAI/bge-reranker-large.

    Returns:
        rerank_tokenizer (XLMRobertaTokenizerFast): The tokenizer of the reranekr model.
        rerank_model (XLMRobertaForSequenceClassification): The loaded reranker model.
    """
    rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
    rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
    _ = rerank_model.eval()
    return rerank_tokenizer, rerank_model

def load_llama2() -> SambaNovaEndpoint:
    """
    Load the LLAMA-2 model from SambaStudio.

    Returns:
        llm (SambaNovaEndpoint): The loaded LLAMA-2 language model.
    """
    base_url='https://sjc3-demo2.sambanova.net'
    project_id='2eeb2d6c-525a-481d-8019-9032f66e5339'
    endpoint_id='e0c392fc-cc71-4161-ab4d-1b7fee26af84'
    api_key='a775c809-c3c8-46eb-9b08-2d4f25349b86'

    llm = SambaNovaEndpoint(
        base_url=base_url,
        project_id=project_id,
        endpoint_id=endpoint_id,
        api_key=api_key,
        model_kwargs={"do_sample": False, "temperature": 0.01, 'max_tokens_to_generate': 512,  
    })
    return llm

def get_embeddings(model_name: str = "intfloat/e5-large-v2") -> HuggingFaceInstructEmbeddings:
    """
    Load embedding model.

    Args:
        model_name (str, optional): The name of the embedding model. Defaults to "intfloat/e5-large-v2".

    Returns:
        HuggingFaceInstructEmbeddings: The loaded embedding model.
    """
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = model_name
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        embed_instruction="",  
        query_instruction="Represent this sentence for searching relevant passages: ",
        encode_kwargs=encode_kwargs)
    return embeddings