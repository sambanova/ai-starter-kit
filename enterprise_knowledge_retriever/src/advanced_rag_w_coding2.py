import yaml
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from utils.rag.rag_components import RAGComponents
from utils.code_gen.codegen_components import CodeGenComponents
from langgraph.graph import END, StateGraph

class CodeRAGGraphState(TypedDict):

    question : str
    subquestions: List[str]
    entities: List[str]
    generation : str
    documents : List[str]
    answers: List[str] 
    original_question: str 
    code: str
    runnable: str
    error: str
    rag_counter: int
    code_counter: int
    examples: Optional[list]

class CodeRAG(RAGComponents, CodeGenComponents):
    
    def __init__(self, 
                 configs: Dict, 
                 embeddings: HuggingFaceInstructEmbeddings, 
                 vectorstore: Chroma, 
                 examples: Optional[list] = None):
        super().__init__(configs=configs, embeddings=embeddings, vectorstore=vectorstore, examples=examples)
        self.app = None 

    def initialize_code_rag(self, state) -> None:
        """
        Initializes the complex RAG with the given state.

        Args:
            state (Dict[str, str]): A dictionary containing the state of the RAG, including the question.

        Returns:
            None
        """

        print("---Initializing---")
        question: str = state["question"]
        print(question)

        return {"rag_counter": 0, "subquestions": [], "answers": [], "examples": self.examples, "original_question": question, "code_counter": 0}

    def create_rag_nodes(self):

        workflow = StateGraph(CodeRAGGraphState)

        # Define the nodes
        workflow.add_node("initialize_code_rag", self.initialize_code_rag)
        workflow.add_node("reformulate_query", self.reformulate_query)
        workflow.add_node("get_new_query", self.pass_state)
        workflow.add_node("generate_subquestions", self.generate_subquestions)
        workflow.add_node("detect_entities", self.detect_entities)
        workflow.add_node("retrieve", self.retrieve_w_filtering)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.rag_generate)
        workflow.add_node("pass_from_qa", self.pass_state)
        workflow.add_node("pass_to_codegen", self.pass_to_codegen)
        workflow.add_node("code_generation", self.code_generation)
        workflow.add_node("determine_runnable_code", self.determine_runnable_code)
        workflow.add_node("refactor_code", self.refactor_code)
        workflow.add_node("code_error_msg", self.code_error_msg)
        workflow.add_node("failure_msg", self.failure_msg)
        workflow.add_node("aggregate_answers", self.aggregate_answers)
        workflow.add_node("return_final_answer", self.final_answer)

        return workflow

    def build_rag_graph(self, workflow):

        # Build graph

        workflow.set_entry_point("initialize_code_rag")
        workflow.add_conditional_edges(
            "initialize_code_rag",
            self.use_examples,
            {
                "answer_generation": "get_new_query",
                "example_selection": "reformulate_query",
            },
        )
        workflow.add_edge("reformulate_query", "get_new_query")
        workflow.add_conditional_edges(
            "get_new_query",
            self.route_question,
            {
                "answer_generation": "detect_entities",
                "subquery_generation": "generate_subquestions",
            },
        )
        workflow.add_edge("generate_subquestions", "detect_entities")
        workflow.add_edge("detect_entities", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.check_hallucinations,
            {
                "not supported": "failure_msg",
                "useful": "pass_from_qa",
                "not useful": "failure_msg",
            }
        )
        workflow.add_edge("failure_msg", "pass_from_qa")
        workflow.add_conditional_edges(
            "pass_from_qa", 
            self.determine_cont, 
            {
                "continue": "pass_to_codegen",
                "iterate": "detect_entities",
            },
        )
        workflow.add_conditional_edges(
            "pass_to_codegen",
            self.route_question_to_code,
            {
                "llm": "aggregate_answers",
                "codegen": "code_generation",
            },
        )
        workflow.add_edge("code_generation", "determine_runnable_code")
        workflow.add_conditional_edges(
        "determine_runnable_code",
        self.decide_to_refactor,
        {
            "executed": "return_final_answer", 
            "exception": "refactor_code",
            "unsuccessful": "code_error_msg"
        },
        )
        workflow.add_edge("refactor_code", "determine_runnable_code")
        workflow.add_edge("code_error_msg", "return_final_answer")
        workflow.add_edge("aggregate_answers", "return_final_answer")
        workflow.add_edge("return_final_answer", END)

        app = workflow.compile()

        return app
    
    def initialize(self):

        self.init_llm()
        self.init_router()
        self.init_reform_chain()
        self.init_subquery_chain()
        self.init_entity_chain()
        self.init_example_judge()
        self.init_retrieval_grader()
        self.init_qa_chain()
        self.init_hallucination_chain()
        self.init_grading_chain()
        self.init_code_router()
        self.init_codegen_chain()
        self.init_codegen_qc_chain()
        self.init_refactor_chain()
        self.init_failure_chain()
        self.init_aggregation_chain()
        self.init_final_generation()
    
    def call_rag(self, app, question: str, kwargs: Dict[str,int] = {"recursion_limit": 50}):
        
        response = {}
        output = app.invoke({"question": question}, kwargs)
        response["answer"] = output["generation"]
        sources = [o for o in output["documents"]]
        response["source_documents"] = sources
        
        return response