import os
import sys
from typing import List, Dict
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import END, StateGraph
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from complex_rag.src.base import BaseComponents


class RAGComponents(BaseComponents):

    def __init__(self, config, prompts_path, embeddings, vectorstore):

        self.qa_chain = None
        self.vectorstore = vectorstore
        self.embeddings = embeddings

        self.configs = self.load_config(config)
        self.prompts_path = prompts_path

    ### RAG Chains

    def init_retrieval_grader(self) -> None:

        retrieval_grader_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-retrieval_grading.yaml"))
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

    def init_qa_chain(self) -> None:

        qa_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-qa.yaml"))
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def init_hallucination_chain(self) -> None:

        hallucination_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-hallucination_detection.yaml"))
        self.hallucination_chain = hallucination_prompt | self.llm | JsonOutputParser()

    def init_grading_chain(self) -> None:

        grading_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-answer_grading.yaml"))
        self.grading_chain = grading_prompt | self.llm | JsonOutputParser()

    def init_failure_chain(self) -> None:

        failure_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-failure_msg.yaml"))
        self.failure_chain = failure_prompt | self.llm | StrOutputParser()

    def init_final_generation(self) -> None:

        final_chain_prompt = load_prompt(os.path.join(self.prompts_path, "llama3-prompt_engineering-final_chain.yaml"))
        self.final_chain = final_chain_prompt | self.llm | StrOutputParser()

    ### RAG functionalities

    def initialize_rag(self, state) -> None:

        print("---Initializing---")
        question = state["question"]
        print(question) 

        return {"answers": [], "original_question": question}
        
    def retrieve(self, state):

        question = state["question"]

        search_kwargs = {"k": self.configs["retrieval"]["k_retrieved_documents"]}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        print("---RETRIEVING FOR QUESTION---")
        print(question)

        documents = retriever.invoke(question)

        return {"documents": documents, "question": question}
    
    def rag_generate(self, state):

        print("---GENERATING---")
        question = state["question"]
        documents = state["documents"]
        answers = state["answers"]

        print("---ANSWERING---")
        print(question)

        docs = self._format_docs(documents)

        print("---DOCS---")
        print("length: ", len(docs))
        print(docs)

        generation = self.qa_chain.invoke({"question": question, "context": docs})

        "---ANSWER---"
        print(generation)
        answers.append(generation)

        return {"generation": generation, "answers": answers}
    
    def grade_documents(self, state):

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']

            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {"documents": filtered_docs, "question": question}
        
    def check_hallucinations(self, state):

        print("---CHECK FOR HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        score = self.hallucination_chain.invoke(
        {"documents": documents, "generation": generation}
    )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.grading_chain.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return "not supported"  
        
    def failure_msg(self, state):

        question = state["question"]

        failure_msg = self.failure_chain.invoke({"question": question})
        
        return {"answers": failure_msg}
        
    def return_final_answer(self, state):

        answers = state["answers"]
        if isinstance(answers, list):
            answers = "\n\n".join(a for a in answers)
        original_question = state["original_question"]

        # print("---ANSWERING---")
        # print(original_question)

        # print("---INTERMEDIATE ANSWERS---")
        # print(answers)

        print("---FINAL ANSWER---")
        final_answer = self.final_chain.invoke({"question": original_question, "answers": answers}) 

        return {"generation": final_answer}
    
class RAGGraphState(TypedDict):

    question : str
    generation : str
    documents : List[str]
    answers: List[str] # Only because of the shared method
    original_question: str # Only because of the shared method
    
class COMPLEXRAG(RAGComponents):

    def create_rag_nodes(self):

        workflow = StateGraph(RAGGraphState)

        # Define the nodes

        workflow.add_node("initialize", self.initialize_rag)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.rag_generate)
        workflow.add_node("failure_msg", self.failure_msg)
        workflow.add_node("return_final_answer", self.return_final_answer)

        return workflow

    def build_rag_graph(self, workflow):

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.check_hallucinations,
            {
                "not supported": "failure_msg",
                "useful": "return_final_answer",
                "not useful": "failure_msg",
            }
        )
        workflow.add_edge("failure_msg", "return_final_answer")
        workflow.add_edge("return_final_answer", END)

        app = workflow.compile()

        return app
    
    def initialize(self):

        self.init_llm()
        self.init_retrieval_grader()
        self.init_qa_chain()
        self.init_retrieval_grader()
        self.init_hallucination_chain()
        self.init_grading_chain()
        self.init_failure_chain()
        self.init_final_generation()
    
    def call_rag(self, app, question: str, kwargs: Dict[str,int] = {"recursion_limit": 50}):
        
        response = {}
        output = app.invoke({"question": question}, kwargs)
        response["answer"] = output["generation"]
        sources = [o for o in output["documents"]]
        response["source_documents"] = sources
        
        return response