from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from utils.rag.rag_components import RAGComponents


class RAGGraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    answers: List[str]
    original_question: str


class RAG(RAGComponents):
    def __init__(self, config_path: str, embeddings: Any, vectorstore: Any) -> None:
        super().__init__(configs=config_path, embeddings=embeddings, vectorstore=vectorstore)
        self.app: Optional[Any] = None

    def create_rag_nodes(self) -> StateGraph:
        workflow = StateGraph(RAGGraphState)

        # Define the nodes
        workflow.add_node('initialize', self.initialize_rag)
        workflow.add_node('retrieve', self.retrieve)
        workflow.add_node('generate', self.rag_generate)
        workflow.add_node('return_final_answer', self.final_answer)

        return workflow

    def build_rag_graph(self, workflow: StateGraph) -> None:
        assert isinstance(workflow, StateGraph)
        workflow.set_entry_point('initialize')
        workflow.add_edge('initialize', 'retrieve')
        workflow.add_edge('retrieve', 'generate')
        workflow.add_edge('generate', 'return_final_answer')
        workflow.add_edge('return_final_answer', END)

        app = workflow.compile()

        self.app = app

    def call_rag(self, question: str, kwargs: Dict[str, int] = {'recursion_limit': 50}) -> Dict[str, Any]:
        response: Dict[str, Any] = {}
        assert self.app is not None
        output = self.app.invoke({'question': question}, kwargs)
        response['answer'] = output['generation']
        sources = [o for o in output['documents']]
        response['source_documents'] = sources

        return response
