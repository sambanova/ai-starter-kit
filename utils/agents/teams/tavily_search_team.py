import os
import sys
from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.graph.graph import CompiledGraph


current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.rag.rag_components import RAGComponents  # type: ignore
from utils.code_gen.codegen_components import CodeGenComponents  # type: ignore
from utils.search.search_components import SearchComponents  # type: ignore


class SearchGraphState(TypedDict):
    """
    Represents the state of a search graph.

    Args:
        question: The original question being asked by the user.
        generation: The current generation from a LLM agent in the pipeline.
        documents: A list of documents after retrieving from the vectorstore.
        answers: A list of answers that have been accumulated from the app.
        original_question: The original question being searched, which may need to bre retrieved/used after 
        query reformulation or subquery decomposition..
    """
    question: str
    generation: str
    documents: List[str]
    answers: List[str]
    original_question: str


class TavilySearchTeam(RAGComponents, CodeGenComponents, SearchComponents):
    """
    A class that combines RAG, CodeGen, and Search components to generate responses as needed.
    """

    def create_search_nodes(self) -> StateGraph:
        """
        Creates the nodes for the TavilySearchTeam graph state.

        Returns:
            The StateGraph object containing the nodes for the TavilySearchTeam graph state.
        """

        workflow: StateGraph = StateGraph(SearchGraphState)

        # Define the nodes
        
        workflow.add_node('search', self.tavily_web_search)
        workflow.add_node('grade_documents', self.grade_documents)
        workflow.add_node('generate', self.rag_generate)
        workflow.add_node('failure_msg', self.failure_msg)
        workflow.add_node('return_final_answer', self.final_answer_search)

        return workflow

    def build_search_graph(self, workflow: StateGraph) -> CompiledGraph:
        """
        Builds a graph for the RAG workflow.

        This method constructs a workflow graph that represents the sequence of tasks
        performed by the RAG system. The graph is used to execute the workflow and
        generate code.

        Args:
            workflow: The workflow object (StateGraph containing nodes) to be modified.

        Returns:
            The compiled application object for static TavilySearchTeam
        """

        checkpointer = MemorySaver()

        workflow.set_entry_point('search')
        workflow.add_edge('search', 'grade_documents')
        workflow.add_edge('grade_documents', 'generate')
        workflow.add_conditional_edges(
            'generate',
            self.check_hallucinations,
            {
                'not supported': 'failure_msg',
                'useful': 'return_final_answer',
                'not useful': 'failure_msg',
            },
        )
        workflow.add_edge('failure_msg', 'return_final_answer')
        workflow.add_edge('return_final_answer', END)

        app: CompiledGraph = workflow.compile(checkpointer=checkpointer)

        return app

    def initialize(self) -> None:
        """
        Initializes all the components of the static TavilySearchTeam app.
        """

        self.init_llm()
        self.init_retrieval_grader()
        self.init_qa_chain()
        self.init_retrieval_grader()
        self.init_hallucination_chain()
        self.init_grading_chain()
        self.init_failure_chain()
        self.init_final_generation()
