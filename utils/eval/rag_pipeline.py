from typing import Dict
from langchain.chains import RetrievalQA
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

class RAGPipeline:
    """RAG Pipeline for answer generation"""

    def __init__(
        self, 
        llm: BaseLLM, 
        vector_db_location: str,
        embeddings: Embeddings,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = Chroma(persist_directory=vector_db_location, embedding_function=embeddings)
        
        prompt_template = """Use the following context to answer the question at the end.
        If the answer is not contained in the context, say "I don't know".

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
        )
    
    def generate(self, query: str) -> Dict:
        """Generate an answer for the given query"""
        response = self.qa_chain.run(query)
        return {"answer": response}