import yaml
from typing import List, Dict
from typing_extensions import TypedDict
from utils.rag.rag_components import RAGComponents
from langgraph.graph import END, StateGraph

class RAGGraphState(TypedDict):

    question : str
    generation : str
    documents : List[str]
    answers: List[str] 
    original_question: str 
    
class RAG(RAGComponents):
    
    def __init__(self, config_path, embeddings, vectorstore):
        config = self.load_config(config_path)
        super().__init__(configs=config, embeddings=embeddings, vectorstore=vectorstore)
        self.app = None 

    def create_rag_nodes(self):

        workflow = StateGraph(RAGGraphState)

        # Define the nodes
        workflow.add_node("initialize", self.initialize_rag)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.rag_generate)
        workflow.add_node("return_final_answer", self.return_final_answer)

        return workflow

    def build_rag_graph(self, workflow):

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "return_final_answer")
        workflow.add_edge("return_final_answer", END)

        app = workflow.compile()

        self.app = app

    def call_rag(self, question: str, kwargs: Dict[str,int] = {"recursion_limit": 50}):
        
        response = {}
        output = self.app.invoke({"question": question}, kwargs)
        response["answer"] = output["generation"]
        sources = [o for o in output["documents"]]
        response["source_documents"] = sources
        
        return response
