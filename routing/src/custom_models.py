import numpy as np
from keybert.backend import BaseEmbedder
from keybert.llm import BaseLLM
from keybert.llm._utils import process_candidate_keywords
from tqdm import tqdm

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        if isinstance(documents, str):
            embeddings = self.embedding_model.embed_query(documents)
        elif isinstance(documents, list):
            embeddings = self.embedding_model.embed_documents(documents)
        elif isinstance(documents, np.ndarray):
            embeddings = self.embedding_model.embed_documents(documents.tolist())
        return np.array(embeddings)
    
DEFAULT_PROMPT = """
<s>[INST]
You are a helpful assistant in retrieving kekwords in documents. 
You have a list of documents. Please find the keywords that are present in the document and separate them with commas. 
Please sort the keywords and return only the top 5 keywords.
Make sure you to only return the keywords and say nothing else. For example, don't say: 
"Here are the keywords present in the document"
[/INST]

Document:
[DOCUMENTS]
</s>
"""

class CustomTextGeneration(BaseLLM):
    def __init__(self, 
                 model, 
                 prompt: str = None,
                 verbose: bool = False
                 ):
        super().__init__()
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.verbose = verbose
    def extract_keywords(self, documents: list[str], candidate_keywords: list[list[str]] = None) -> list:
        """ Extract topics

        Arguments:
            documents: The documents to extract keywords from
            candidate_keywords: A list of candidate keywords that the LLM will fine-tune
                        For example, it will create a nicer representation of
                        the candidate keywords, remove redundant keywords, or
                        shorten them depending on the input prompt.

        Returns:
            list: All keywords for each document
        """
        all_keywords = []
        candidate_keywords = process_candidate_keywords(documents, candidate_keywords)

        for document, candidates in tqdm(zip(documents, candidate_keywords), disable=not self.verbose):
            prompt = self.prompt.replace("[DOCUMENTS]", document)
            if candidates is not None:
                prompt = prompt.replace("[CANDIDATES]", ", ".join(candidates))
            # Extract result from generator and use that as label
            keywords = self.model(prompt).replace(prompt, "")
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords