"""
Bulk Question-Answering Evaluation Script

This standalone script enables automated bulk testing of question-answering capabilities
using a pre-built vector database and an Excel file containing questions. It's designed
for quick evaluation of RAG (Retrieval-Augmented Generation) performance across multiple
questions.

Key Features:
- Processes questions in bulk from an Excel file
- Uses a pre-built vector database for document retrieval
- Measures and records performance metrics including:
  * Preprocessing time
  * LLM inference time
  * Token count and tokens per second
  * Source documents used for each answer
- Saves results incrementally to prevent data loss
- Supports resuming interrupted runs by skipping already answered questions

Input Requirements:
- Vector database path containing stored documents for RAG
- Excel file (.xlsx) with questions in a column named 'Questions'

Output:
- Creates a new Excel file with '_output' suffix containing:
  * Original questions
  * Generated answers
  * Source documents used
  * Performance metrics for each response

Usage:
    python bulkQA.py <vectordb_path> <questions_path>

Example:
    python bulkQA.py ./path/to/vectordb ./path/to/questions.xlsx
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from transformers import AutoTokenizer, PreTrainedTokenizerBase

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval, RetrievalQAChain, load_chat_prompt

sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')


class TimedRetrievalQAChain(RetrievalQAChain):
    # override call method to return times
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        response: Dict[str, Any] = {}
        start_time = time.time()
        documents = self.retriever.invoke(inputs['question'])
        if self.rerank:
            documents = self.rerank_docs(inputs['question'], documents, self.final_k_retrieved_documents)
        docs = self._format_docs(documents)
        end_preprocessing_time = time.time()
        response['answer'] = qa_chain.invoke({'question': inputs['question'], 'context': docs})
        end_llm_time = time.time()
        response['source_documents'] = documents
        response['start_time'] = start_time
        response['end_preprocessing_time'] = end_preprocessing_time
        response['end_llm_time'] = end_llm_time
        return response


def analyze_times(
    answer: str,
    start_time: float,
    end_preprocessing_time: float,
    end_llm_time: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, float | int]:
    preprocessing_time = end_preprocessing_time - start_time
    llm_time = end_llm_time - end_preprocessing_time
    token_count = len(tokenizer.encode(answer))
    tokens_per_second = token_count / llm_time
    perf = {
        'preprocessing_time': preprocessing_time,
        'llm_time': llm_time,
        'token_count': token_count,
        'tokens_per_second': tokens_per_second,
    }
    return perf


def generate(
    qa_chain: RetrievalQAChain, question: str, tokenizer: PreTrainedTokenizerBase
) -> Tuple[str, Set[str], Dict[str, float | int]]:
    response = qa_chain.invoke({'question': question})
    answer = response.get('answer')
    sources = set([f'{sd.metadata["filename"]}' for sd in response['source_documents']])
    start_time = response.get('start_time')
    end_preprocessing_time = response.get('end_preprocessing_time')
    end_llm_time = response.get('end_llm_time')
    assert (
        isinstance(answer, str)
        and isinstance(start_time, float)
        and isinstance(end_preprocessing_time, float)
        and isinstance(end_llm_time, float)
    )
    times = analyze_times(
        answer,
        start_time,
        end_preprocessing_time,
        end_llm_time,
        tokenizer,
    )
    return answer, sources, times


def process_bulk_QA(vectordb_path: str, questions_file_path: str) -> str:
    documentRetrieval = DocumentRetrieval(sambanova_api_key=sambanova_api_key)
    tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')  # type: ignore[no-untyped-call]
    if os.path.exists(vectordb_path):
        # load the vectorstore
        embeddings = documentRetrieval.load_embedding_model()
        vectorstore = documentRetrieval.load_vdb(vectordb_path, embeddings, collection_name='ekr_default_collection')
        print('Database loaded')
        documentRetrieval.init_retriever(vectorstore)
        print('retriever initialized')
        # get qa chain
        assert isinstance(documentRetrieval.retriever, BaseRetriever)
        qa_chain = TimedRetrievalQAChain(
            retriever=documentRetrieval.retriever,
            llm=documentRetrieval.llm,
            qa_prompt=load_chat_prompt(os.path.join(repo_dir, documentRetrieval.prompts['qa_prompt'])),
            rerank=documentRetrieval.retrieval_info['rerank'],
            final_k_retrieved_documents=documentRetrieval.retrieval_info['final_k_retrieved_documents'],
            conversational=False,
        )
    else:
        raise FileNotFoundError(f'vector db path {vectordb_path} does not exist')
    if os.path.exists(questions_file_path):
        df = pd.read_excel(questions_file_path)
        print(df)
        output_file_path = questions_file_path.replace('.xlsx', '_output.xlsx')
        if 'Answer' not in df.columns:
            df['Answer'] = ''
            df['Sources'] = ''
            df['preprocessing_time'] = ''
            df['llm_time'] = ''
            df['token_count'] = ''
            df['tokens_per_second'] = ''
        for index, row in df.iterrows():
            if row['Answer'].strip() == '':  # Only process if 'Answer' is empty
                try:
                    # Generate the answer
                    print(f'Generating answer for row {index}')
                    answer, sources, times = generate(qa_chain, row['Questions'], tokenizer)
                    df.at[index, 'Answer'] = answer
                    df.at[index, 'Sources'] = sources
                    df.at[index, 'preprocessing_time'] = times.get('preprocessing_time')
                    df.at[index, 'llm_time'] = times.get('llm_time')
                    df.at[index, 'token_count'] = times.get('token_count')
                    df.at[index, 'tokens_per_second'] = times.get('tokens_per_second')
                except Exception as e:
                    print(f'Error processing row {index}: {e}')
                # Save the file after each iteration to avoid data loss
                df.to_excel(output_file_path, index=False)
            else:
                print(f"Skipping row {index} because 'Answer' is already in the document")
        return output_file_path
    else:
        raise f'questions file path {questions_file_path} does not exist'


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="""use a vectordb and an excel file with questions in the first column and generate answers
         for all the questions"""
    )
    parser.add_argument('vectordb_path', type=str, help='vector db path with stored documents for RAG')
    parser.add_argument('questions_path', type=str, help='xlsx file containing questions in a column named Questions')
    args = parser.parse_args()
    # process in bulk
    out_file = process_bulk_QA(args.vectordb_path, args.questions_path)
    print(f'Finished, responses in: {out_file}')
