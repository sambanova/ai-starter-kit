# Import libraries
import os
import sys
import shutil
import time

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, "..")) # absolute path for ekr_rag directory
repo_dir = os.path.abspath(os.path.join(kit_dir, "..")) # absolute path for starter-kit directory
print('kit_dir: %s'%kit_dir)
print('repo_dir: %s'%repo_dir)

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from src.document_retrieval import DocumentRetrieval
from utils.parsing.sambaparse import SambaParse, parse_doc_universal

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,f"data/my-vector-db")

time_start = time.time()

# Initialize DocumentRetrieval class
documentRetrieval =  DocumentRetrieval()

# Parse the documents and get the text chunks
additional_metadata = {}
_, _, text_chunks = parse_doc_universal(doc=kit_dir+'/data/test', additional_metadata=additional_metadata)
print('Nb of chunks: %d'%len(text_chunks))

# Create vector store
embeddings = documentRetrieval.load_embedding_model()
if os.path.exists(PERSIST_DIRECTORY):
    shutil.rmtree(PERSIST_DIRECTORY)
    print(f"The directory Chroma has been deleted.")
#vectorstore = documentRetrieval.create_vector_store(text_chunks, embeddings, output_db=None)
vectorstore = documentRetrieval.create_vector_store(text_chunks, embeddings, output_db=PERSIST_DIRECTORY)

# Create conversation chain
documentRetrieval.init_retriever(vectorstore)
conversation = documentRetrieval.get_qa_retrieval_chain()

# Ask questions about your data
user_question = "What is a composition of experts?"

response = conversation.invoke({"question":user_question})
print(response['question'])
print(response['answer'])

time_end = time.time()

# Assertions
assert 'source_documents' in response, "The response should have a 'source_documents' key."
assert len(response['source_documents']) >= 1, "There should be at least one source chunk."
assert 'answer' in response, "The response should have an 'answer' key."
assert response['answer'], "The response should not be empty."
print("Test passed")
print("Time: %g s"%(time_end - time_start))