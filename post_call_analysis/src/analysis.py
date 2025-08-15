import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(repo_dir, '.env'))

import concurrent.futures
from typing import Any, Dict, List, Tuple

import yaml
from langchain.chains import LLMChain, ReduceDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    OutputFixingParser,
    ResponseSchema,
    StructuredOutputParser,
)
from langchain.prompts import load_prompt
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser

from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
llm_info = config['llm']
retrieval_info = config['retrieval']
embedding_model_info = config['embedding_model']
model = APIGateway.load_llm(
    type=llm_info['api'],
    streaming=True,
    bundle=llm_info['bundle'],
    do_sample=llm_info['do_sample'],
    max_tokens_to_generate=llm_info['max_tokens_to_generate'],
    temperature=llm_info['temperature'],
    model=llm_info['model'],
    process_prompt=False,
)


def load_conversation(transcription: str, transcription_path: str) -> List[Document]:
    """Load a conversation as langchain Document

    Args:
        transcription (str): The transcription of the conversation.
        transcription_path (str): The path of the transcription file.

    Returns:
        List[Document]: The conversation as a list of Documents.
    """
    doc = Document(page_content=transcription, metadata={'source': transcription_path})
    return [doc]


def reduce_call(conversation: List[Document]) -> Any:
    """
    Reduce the conversation by applying the ReduceDocumentsChain.

    Args:
        conversation (List[Document]): The conversation to reduce.

    Returns:
        str: The reduced conversation.
    """
    reduce_prompt = load_prompt(os.path.join(kit_dir, 'prompts/reduce.yaml'))
    reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name='transcription_chunks')
    # Combines and iteratively reduces the documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=1200,
    )
    print('reducing call')
    new_document = reduce_documents_chain.invoke(conversation)['output_text']
    print('call reduced')
    return new_document


def get_summary(conversation: str, model: LLM = model) -> str:
    """
    Summarizes a conversation.

    Args:
        conversation (str): The conversation to summarize.
        model (Langchain LLM Model, optional): The language model to use for summarization.
            Defaults to a SambaStudio model.

    Returns:
        str: The summary of the conversation.
    """
    summarization_prompt = load_prompt(os.path.join(kit_dir, 'prompts/summarization.yaml'))
    output_parser = StrOutputParser()
    summarization_chain = summarization_prompt | model | output_parser
    input_variables = {'conversation': conversation}
    print('summarizing')
    summarization_response = summarization_chain.invoke(input_variables)
    print('summarizing done')
    return summarization_response


def classify_main_topic(conversation: str, classes: List[str], model: LLM = model) -> List[str]:
    """
    Classify the topic of a conversation.

    Args:
        conversation (str): The conversation to classify.
        classes (List[str]): The list of classes to classify the conversation into.
        model (Langchain LLM Model, optional): The language model to use for classification. Defaults to a SambaStudio
        model.

    Returns:
        List[str]: The list of classes that the conversation was classified into.
    """
    topic_classification_prompt = load_prompt(os.path.join(kit_dir, 'prompts/topic_classification.yaml'))
    list_output_parser = CommaSeparatedListOutputParser()
    list_format_instructions = list_output_parser.get_format_instructions()
    list_fixing_output_parser = OutputFixingParser.from_llm(parser=list_output_parser, llm=model)
    topic_classification_chain = topic_classification_prompt | model | list_fixing_output_parser
    input_variables = {
        'conversation': conversation,
        'topic_classes': '\n\t- '.join(classes),
        'format_instructions': list_format_instructions,
    }
    print('classification')
    topic_classification_response = topic_classification_chain.invoke(input_variables)
    print('classification done')
    return topic_classification_response


def get_entities(conversation: str, entities: List[str], model: LLM = model) -> Dict[str, Any]:
    """
    Extract entities from a conversation.

    Args:
        conversation (str): The conversation to extract entities from.
        entities (List[str]): The list of entities to extract.
        model (Langchain LLM Model, optional): The LLM model to use for extraction. Defaults to a SambaStudio model.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted entities.
            The keys are the entity names, and the values are the extracted entities.
    """
    ner_prompt = load_prompt(os.path.join(kit_dir, 'prompts/ner.yaml'))
    response_schemas = []
    for entity in entities:
        response_schemas.append(ResponseSchema(name=entity, description=f'{entity}s find in conversation', type='list'))
    entities_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    entities_fixing_output_parser = OutputFixingParser.from_llm(parser=entities_output_parser, llm=model)
    ner_chain = ner_prompt | model | entities_fixing_output_parser
    input_variables = {
        'conversation': conversation,
        'entities': '\n\t- '.join(entities),
        'format_instructions': entities_output_parser.get_format_instructions(),
    }
    print('extracting entities')
    ner_response = ner_chain.invoke(input_variables)
    print('extracting entities done')
    return ner_response


def get_sentiment(conversation: str, sentiments: List[Any], model: LLM = model) -> str:
    """
    get the overall sentiment of the user in a conversation.

    Args:
        conversation (str): The conversation to analyse.
        setiments (list): Listo of posible sentioments to clasiffy
        model (Langchain LLM Model, optional): The language model to use for summarization. Defaults to a SambaStudio
        model.

    Returns:
        str: The overall sentiment of the user.
    """
    sentiment_analysis_prompt = load_prompt(os.path.join(kit_dir, 'prompts/sentiment_analysis.yaml'))
    list_output_parser = CommaSeparatedListOutputParser()
    list_format_instructions = list_output_parser.get_format_instructions()
    list_fixing_output_parser = OutputFixingParser.from_llm(parser=list_output_parser, llm=model)
    sentiment_analysis_chain = sentiment_analysis_prompt | model | list_fixing_output_parser
    input_variables = {
        'conversation': conversation,
        'sentiments': sentiments,
        'format_instructions': list_format_instructions,
    }
    print('sentiment analysis')
    sentiment_analysis_response = sentiment_analysis_chain.invoke(input_variables)
    print('sentiment analysis done')
    return sentiment_analysis_response[0]


def get_nps(conversation: str, model: LLM = model) -> Dict[str, Any]:
    """get a prediction of a possible net promoter score for a given conversation

    Args:
        conversation (str): The conversation to analyse.
        model (Langchain LLM Model, optional): The language model to use for summarization.
            Defaults to a SambaStudio model.

    Returns:
        nps (Dict): description of the predicted score and the corresponding score
    """
    nps_response_schemas = [
        ResponseSchema(name='description', description='reasoning', type='str'),
        ResponseSchema(name='score', description='punctuation from 1 to 10 of the NPS', type='int'),
    ]
    nps_output_parser = StructuredOutputParser.from_response_schemas(nps_response_schemas)
    format_instructions = nps_output_parser.get_format_instructions()
    nps_fixing_output_parser = OutputFixingParser.from_llm(parser=nps_output_parser, llm=model)
    nps_prompt = load_prompt(os.path.join(kit_dir, 'prompts/nps.yaml'))
    nps_chain = nps_prompt | model | nps_fixing_output_parser
    input_variables = {'conversation': conversation, 'format_instructions': format_instructions}
    print(f'predicting nps')
    nps = nps_chain.invoke(input_variables)
    print(f'nps chain finished')
    return nps


def get_call_quality_assessment(
    conversation: str, factual_result: Dict[str, Any], procedures_result: Dict[str, Any]
) -> Tuple[float, Any, Any]:
    """
    Return the calculated quality assessment of the given conversation.

    Args:
        conversation (str): The conversation to analyse.
        factual_result (Dict): The factual analysis result of the conversation.
        procedures_result (Dict): The procedures analysis result of the conversation.

    Returns:assessment
        float: The calculated quality assessment of the given conversation.
    """
    # initialize the score
    total_score = 0
    # predict a NPS of the call
    nps = get_nps(conversation)
    total_score += nps['score'] * 10
    # include the factual analysis score
    total_score += factual_result['score']
    # include the procedures analysis score
    if len(procedures_result['evaluation']) == 0:
        total_score += 100
    else:
        total_score += procedures_result['evaluation'].count(True) / len(procedures_result['evaluation']) * 100
    # Simple average
    overall_score = total_score / 3
    return overall_score, nps['description'], nps['score']


def set_retriever(documents_path: str, urls: List[str]) -> Any:
    """
    Set up a Faiss vector database for document retrieval.

    Args:
        documents_path (str): The path to the directory containing the documents.
        urls (List[str]: The list of Urls to scrape and load)

    Returns:
        langchain retriever: The Faiss retriever to be used whit lang chain retrieval chains.
    """
    print('setting retriever')
    vectordb = VectorDb()

    retriever = vectordb.create_vdb(
        documents_path,
        retrieval_info['chunk_size'],
        retrieval_info['chunk_overlap'],
        retrieval_info['db_type'],
        None,
        load_txt=True,
        load_pdf=True,
        urls=urls,
        embedding_type=embedding_model_info.get('type'),
        batch_size=embedding_model_info.get('batch_size'),
        bundle=embedding_model_info.get('bundle'),
        model=embedding_model_info.get('model'),
    ).as_retriever()

    print('retriever set')
    return retriever


def factual_accuracy_analysis(conversation: str, retriever: Any, model: LLM = model) -> Dict[str, Any]:
    """
    Analyse the factual accuracy of the given conversation.

    Args:
        conversation (str): The conversation to analyse.
        retriever (langchain Retriever): The langchain retriever to use for document similarity retrieval.
        model (Langchain LLM Model, optional): The language model to use for summarization and classification.
            Defaults to a SambaStudio model.

    Returns:
        dict: A dictionary containing the factual accuracy analysis results. The keys are:
            - "correct": A boolean indicating whether the provided information is correct.
            - "errors": A list of summarized errors made by the agent, if any.
            - "score": A score from 1 to 100 indicating the overall quality of the agent's response.
    """
    factual_accuracy_analysis_response_schemas = [
        ResponseSchema(name='correct', description='wether or not the provided information is correct', type='bool'),
        ResponseSchema(
            name='errors',
            description='list of summarized errors made by the agent, if there is no errors, empty list',
            type='list',
        ),
        ResponseSchema(
            name='score', description='punctuation from 0 to 100 of the overall quality of the agent', type='int'
        ),
    ]
    factual_accuracy_analysis_output_parser = StructuredOutputParser.from_response_schemas(
        factual_accuracy_analysis_response_schemas
    )
    format_instructions = factual_accuracy_analysis_output_parser.get_format_instructions()
    factual_accuracy_analysis_fixing_output_parser = OutputFixingParser.from_llm(
        parser=factual_accuracy_analysis_output_parser, llm=model
    )
    retrieval_qa_chat_prompt = load_prompt(os.path.join(kit_dir, 'prompts/factual_accuracy_analysis.yaml'))
    combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    input_variables = {'input': conversation, 'format_instructions': format_instructions}
    model_response = retrieval_chain.invoke(input_variables)['answer']
    print('factual check')
    factual_accuracy_analysis_response = factual_accuracy_analysis_fixing_output_parser.invoke(model_response)
    print('factual check done')
    return factual_accuracy_analysis_response


def procedural_accuracy_analysis(conversation: str, procedures_path: str, model: LLM = model) -> Dict[str, Any]:
    """
    Analyse the procedural accuracy of the given conversation.

    Args:
        conversation (str): The conversation to analyse.
        procedures_path (str): The path to the file containing the procedures.
        model (Langchain LLM Model, optional): The language model to use for summarization and classification.
            Defaults to a SambaNovaEndpoint model.
    Returns:
        dict: A dictionary containing the procedural accuracy analysis results. The keys are:
            - "correct": A boolean indicating whether the agent followed all the procedures.
            - "errors": A list of summarized errors made by the agent, if any.
            - "evaluation": A list of booleans evaluating if the agent followed each one of the procedures listed.
    """
    procedures_analysis_response_schemas = [
        ResponseSchema(name='correct', description='wether or not the agent followed all the procedures', type='bool'),
        ResponseSchema(
            name='errors',
            description='list of summarized errors made by the agent, if there is no errors, empty list',
            type='list',
        ),
        ResponseSchema(
            name='evaluation',
            description='list of booleans evaluating if the agent followed each one of the procedures listed',
            type='list[bool]',
        ),
    ]
    procedures_analysis_output_parser = StructuredOutputParser.from_response_schemas(
        procedures_analysis_response_schemas
    )
    format_instructions = procedures_analysis_output_parser.get_format_instructions()
    procedures_analysis_fixing_output_parser = OutputFixingParser.from_llm(
        parser=procedures_analysis_output_parser, llm=model
    )
    procedures_prompt = load_prompt(os.path.join(kit_dir, 'prompts/procedures_analysis.yaml'))
    with open(procedures_path, 'r') as file:
        procedures = file.readlines()
    procedures_chain = procedures_prompt | model | procedures_analysis_fixing_output_parser
    input_variables = {'input': conversation, 'procedures': procedures, 'format_instructions': format_instructions}
    print('proceduress check')
    procedures_analysis_response = procedures_chain.invoke(input_variables)
    print('proceduress check done')
    return procedures_analysis_response


def get_chunks(documents: List[Document]) -> List[Document]:
    """
    Split document in smaller documents.

    Args:
        documents List[Document]: The documents to split.

    Returns:
        documents List[Document]: The splitted documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_documents(documents)


def call_analysis_parallel(
    conversation: List[Document],
    documents_path: str,
    facts_urls: List[str],
    procedures_path: str,
    classes_list: List[str],
    entities_list: List[str],
    sentiment_list: List[str],
) -> Dict[str, Any]:
    """
    Runs analysis steps in parallel.

    Args:
        conversation (str): The conversation to analyse.
        documents_path (str): The path to the directory containing the fact or procedure documents.
        procedures_path (str): The path to the file containing the procedures.
        facts_urls (List[str]): The list of URL to load facts from
        classes_list (List[str]): The list of classes to classify the conversation into.
        entities_list (List[str]): The list of entities to extract.
        sentiment_list (List[str]): The list of sentiments to analyse.

    Returns:
        dict: A dictionary containing the analysis results. The keys are:
            - "summary": The summary of the conversation.
            - "classification": The classes that the conversation was classified into.
            - "entities": The extracted entities.
            - "sentiment": The overall sentiment of the user.
            - "factual_analysis": The factual accuracy analysis results.
            - "procedural_analysis": The procedures accuracy analysis results.
            - "quality_score": A score from 1 to 100 indicating the overall quality of the agent's response.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submitting tasks to executor
        reduced_conversation_future = executor.submit(reduce_call, conversation=conversation)
        retriever = set_retriever(documents_path=documents_path, urls=facts_urls)
        reduced_conversation = reduced_conversation_future.result()
        summary_future = executor.submit(get_summary, conversation=reduced_conversation)
        classification_future = executor.submit(
            classify_main_topic, conversation=reduced_conversation, classes=classes_list
        )
        entities_future = executor.submit(get_entities, conversation=reduced_conversation, entities=entities_list)
        sentiment_future = executor.submit(get_sentiment, conversation=reduced_conversation, sentiments=sentiment_list)
        factual_analysis_future = executor.submit(
            factual_accuracy_analysis, conversation=reduced_conversation, retriever=retriever
        )
        procedural_analysis_future = executor.submit(
            procedural_accuracy_analysis, conversation=reduced_conversation, procedures_path=procedures_path
        )

        # Retrieving results
        summary = summary_future.result()
        classification = classification_future.result()
        entities = entities_future.result()
        sentiment = sentiment_future.result()
        factual_analysis = factual_analysis_future.result()
        procedural_analysis = procedural_analysis_future.result()
    quality_score, nps_analysis, nps_score = get_call_quality_assessment(
        reduced_conversation, factual_analysis, procedural_analysis
    )

    return {
        'summary': summary,
        'classification': classification,
        'entities': entities,
        'sentiment': sentiment,
        'factual_analysis': factual_analysis,
        'procedural_analysis': procedural_analysis,
        'nps_analysis': nps_analysis,
        'nps_score': nps_score,
        'quality_score': quality_score,
    }
