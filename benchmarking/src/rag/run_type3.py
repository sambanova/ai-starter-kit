#!/usr/bin/env python

import sys
import yaml 
sys.path.append("../../")
sys.path.append("../../../")

from RAG_pipeline import *
import utils

def answer_question(query: str, 
                    rerank_model, 
                    rerank_tokenizer, 
                    rag_pipeline: RAGPipeline) -> str:
    """
    Generate output from llm

    Args:
        query (str): The user query
        rerank_model (XLMRobertaForSequenceClassification): The loaded reranker model.
        rerank_tokenizer (XLMRobertaTokenizerFast): The tokenizer of the reranekr model.
        rag_pipeline (RAGPipeline): The loaded RAG pipeline instance.

    Returns:
        final_answer (str): The answer from llm.
    """
    
    device_name = utils.get_device_name(query)[0]
    cur_filter = {"device_name":device_name.lower()}
    try: 
        response = rag_pipeline.qa_chain({"question": query, 
                            "filter": cur_filter, 
                            "reranker": rerank_model, 
                            "tokenizer": rerank_tokenizer,
                            "final_k": 3,
                            })

        final_answer = rag_pipeline.OutputParser(response["answer"])
        sources = response["source_documents"]
    except:
        final_answer = "I don't know the answer."
        sources = ""
    return final_answer, sources

def initialization_type3():
    """
    Initialization function. 

    Returns:
        config (dict): The config file. 
        llm (SambaNovaEndpoint): The LLM model used for generating answers (e.g., Llama, GPT).
        vector_db_location (str): Location of the vector database used for document retrieval.
        embedding_model (HuggingFaceInstructEmbeddings): The embedding model used for encoding documents.
        rerank_tokenizer (XLMRobertaTokenizerFast): The tokenizer of the reranekr model.
        rerank_model (XLMRobertaForSequenceClassification): The loaded reranker model.
        rag_pipeline (RAGPipeline): The pipeline for RAG
        data (dict): The user query database
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs/config_type3.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    vector_db_location = os.path.join(script_dir, config['vector_db_location'])
    embed_model_name = config['embed_model_name'] 

    llm = load_llama2()
    embedding_model = get_embeddings(embed_model_name)
    rerank_tokenizer, rerank_model = load_reranker_model()

    k = config['k'] # number of docs to retrieve
    rag_pipeline = RAGPipeline(llm, vector_db_location, embedding_model, k)

    # load user query database; skip it for the app
    data = utils.get_data_handler_group3()

    return config, llm, vector_db_location, embedding_model, rerank_tokenizer, rerank_model, rag_pipeline, data

if __name__ == "__main__":
    # load initialization
    config, llm, vector_db_location, embedding_model, rerank_tokenizer, rerank_model, rag_pipeline, data = initialization_type3()

    # user query
    ID = 10
    query = [item['query'] for item in data if item['ID'] == ID][0]
    print(f'Question: {query}')

    # generate output
    final_answer, sources = answer_question(query, rerank_model, rerank_tokenizer, rag_pipeline)
    print(f'Response: {final_answer}')
