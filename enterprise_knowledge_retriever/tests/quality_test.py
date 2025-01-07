import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
logger.info(f'kit_dir: {kit_dir}')
logger.info(f'repo_dir: {repo_dir}')

sys.path.append(kit_dir)
sys.path.append(repo_dir)


import asyncio

import weave
import yaml

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from utils.eval.dataset import WeaveDatasetManager
from utils.eval.models import CorrectnessLLMJudge, WeaveRAGModel

PERSIST_DIRECTORY = os.path.join(kit_dir, 'tests', 'vectordata', 'my-vector-db')
TEST_DATA_PATH = os.path.join(kit_dir, 'tests', 'data', 'test')
CONFIG_PATH = os.path.join(repo_dir, 'utils', 'eval', 'config.yaml')


sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


document_retrieval = DocumentRetrieval(sambanova_api_key=sambanova_api_key)

text_chunks = document_retrieval.parse_doc(doc_folder=TEST_DATA_PATH)

embeddings = document_retrieval.load_embedding_model()

vectorstore = document_retrieval.create_vector_store(text_chunks, embeddings, output_db=PERSIST_DIRECTORY)

document_retrieval.init_retriever(vectorstore)

conversation_chain = document_retrieval.get_qa_retrieval_chain()

judge_info = config['eval_llm']

rag_info = config['rag']['llm']

judge = CorrectnessLLMJudge(**judge_info)
rag_model = WeaveRAGModel(**rag_info)

data_manager = WeaveDatasetManager()

data = data_manager.create_raw_dataset(os.path.join(kit_dir, 'tests', 'data', 'rag_data.csv'))

for i in range(len(data)):
    response = conversation_chain.invoke({'question': data[i].get('query', '')})
    data[i]['context'] = response.get('source_documents', '')
    data[i]['completion'] = response.get('answer', '')

evaluation = weave.Evaluation(name=' '.join(str(value) for value in judge_info.values()), dataset=data, scorers=[judge])

evaluation_results = asyncio.run(evaluation.evaluate(rag_model))
