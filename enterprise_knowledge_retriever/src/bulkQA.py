import os
import sys
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
 
CONFIG_PATH = os.path.join(kit_dir,'config.yaml')

def generate(qa_chain, question):
    response = qa_chain.invoke({"question": question})
    answer =  response.get('answer')
    sources = set([
            f'{sd.metadata["filename"]}'
            for sd in response["source_documents"]
        ])
    return answer, sources

def process_bulk_QA(vectordb_path, questions_file_path):
    documentRetrieval =  DocumentRetrieval()
    if os.path.exists(vectordb_path):
        # load the vectorstore
        embeddings = documentRetrieval.load_embedding_model()
        vectorstore = documentRetrieval.load_vdb(vectordb_path, embeddings)
        print("Database loaded")
        documentRetrieval.init_retriever(vectorstore)
        print("retriever initialized")
        #get qa chain
        qa_chain = documentRetrieval.get_qa_retrieval_chain()
    else:
        raise f"vector db path {vectordb_path} does not exist"
    if os.path.exists(questions_file_path):
        df = pd.read_excel(questions_file_path)
        print(df)
        output_file_path = questions_file_path.replace('.xlsx', '_output.xlsx')
        if 'Answer' not in df.columns:
            df['Answer'] = ''
            df['Sources'] = ''
        for index, row in df.iterrows():
            if row['Answer'].strip()=='':  # Only process if 'Answer' is empty
                try:
                    # Generate the answer
                    print(f"Generating answer for row {index}")
                    answer, sources = generate(qa_chain, row['Questions'])
                    df.at[index, 'Answer'] = answer
                    df.at[index, 'Sources'] = sources
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                # Save the file after each iteration to avoid data loss
                df.to_excel(output_file_path, index=False)
            else:
                print(f"Skipping row {index} because 'Answer' is already in the document")
        return output_file_path
    else:
        raise f"questions file path {questions_file_path} does not exist"
                                                      
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='use a vectordb and an excel file with questions in the first column and generate answers for all the questions')
    parser.add_argument('vectordb_path', type=str, help='vector db path with stored documents for RAG')
    parser.add_argument('questions_path', type=str, help='xlsx file containing questions in a column named Questions')
    args = parser.parse_args()
    # process in bulk
    out_file = process_bulk_QA(args.vectordb_path, args.questions_path)
    print(f"Finished, responses in: {out_file}")