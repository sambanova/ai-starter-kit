from typing import Dict

from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


class RAGPipeline:
    """RAG Pipeline for answer generation"""

    def __init__(
        self,
        llm: BaseLLM,
        vector_db_location: str,
        embeddings: Embeddings,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = Chroma(persist_directory=vector_db_location, embedding_function=embeddings)

        print(f'This is the vector db {vector_db_location}')

        prompt_template = """
        <|begin_of_text|>
        <|start_header_id|>
        system
        <|end_header_id|>
        You are a helpful, respectful and honest assistant designated answer
        questions related to the user's document.If the user tries to ask out of 
        topic questions do not engange in the conversation.If the given context 
        is not sufficient to answer the question,Do not answer the question.
        <|eot_id|>
        <|start_header_id|>
        user
        <|end_header_id|>
        Answer the user question based on the context provided below
        Context :{context}
        Question: {question}
        <|eot_id|>
        <|start_header_id|>
        assistant
        <|end_header_id|>"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={'prompt': PROMPT},
            return_source_documents=True,
        )

    def generate(self, query: str) -> Dict:
        docs = self.vector_store.similarity_search(query)
        print(docs)
        # print(docs[0].page_content)
        """Generate an answer for the given query"""
        response = self.qa_chain.invoke(query)
        return {'answer': response}
