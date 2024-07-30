import os
import sys
from typing import Any, Dict, List, TypedDict, Annotated
import operator
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.graph import CompiledGraph

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.rag.base_components import BaseComponents  # type: ignore
from utils.agents.supervisor import SupervisorComponents  # type: ignore


class SuperState(TypedDict):
    """
    A dictionary representing the current state of the supervisor.

    Args:
        question: The user question being asked.
        query_history: A list of previous queries made by the user.
        answer_history: A list of previous answers given by the team.
        teams: A list of team names.
        state: A dictionary containing the current state..
        next: The next action to take.
    """

    question: str
    query_history: Annotated[List[str], operator.add]
    answer_history: Annotated[List[str], operator.add]
    teams: List[str]
    state: dict
    next: str


class CRAGSupervisor(SupervisorComponents):
    def create_supervisor(self) -> CompiledGraph:
        """
        Creates a compiled supervisor graph/app for the corrective RAG team.

        Returns:
            The compiled supervisor graph/app.
        """

        checkpointer = MemorySaver()

        sup_graph: StateGraph = StateGraph(SuperState)
        sup_graph.add_node('super_router', self.supervisor_router)
        sup_graph.add_edge('super_router', END)
        sup_graph.set_entry_point('super_router')
        sup_graph_compiled: CompiledGraph = sup_graph.compile(checkpointer=checkpointer)

        return sup_graph_compiled

    def initialize(self) -> None:
        """
        Initializes all the components of the supervisor app.
        """

        self.init_llm()
        self.init_supervisor_router()


class TeamRAGGraphState(TypedDict):
    """
    Represents the state of a team's RAG graph.

    Args:
        question: The user question being asked.
        generation: The most recent generation of the team.
        documents: A list of documents retrieved from the vectorstore.
        answers: A list of answers provided by the team.
        original_question: The original question being asked, in case of subquery generation, query reformulation, etc.
        next: The next step in the pipeline.
    """

    question: str
    generation: str
    documents: List[str]
    answers: List[str]
    original_question: str
    next: str


class TeamCRAG(BaseComponents):
    """
    A hiearchical agentic team, with a supervisor, Corrective RAG and corrective
    Tavily search team.
    """

    def __init__(
        self,
        supervisor_app: CompiledGraph,
        rag_app: CompiledGraph,
        search_app: CompiledGraph,
        return_app: CompiledGraph,
    ) -> None:
        """
        Initializes the Corrective RAG Team.

        Args:
            supervisor_app: The supervisor application.
            rag_app: The RAG application.
            search_app: The search application.
            return_app: The return message application.
        """

        self.team_app = StateGraph(TeamRAGGraphState)
        self.supervisor_app = supervisor_app
        self.rag_app = rag_app
        self.search_app = search_app
        self.return_app = return_app

    def create_team_graph(self) -> None:
        """
        Creates the nodes for the TeamCRAG graph state.

        Returns:
            The StateGraph object containing the nodes for the TeamCRAG graph state.
        """

        self.team_app.add_node('supervisor', self.supervisor_app)
        self.team_app.add_node('rag', self.rag_app)
        self.team_app.add_node('search', self.search_app)
        self.team_app.add_node('return_message', self.return_app)

    def build_team_graph(self) -> CompiledGraph:
        """
        Builds a graph for the TeamCRAG workflow.

        This method constructs a workflow graph that represents the sequence of tasks
        performed by the TeamCRAG system.

        Args:
            workflow: The workflow object (StateGraph containing nodes) to be modified.

        Returns:
            The compiled application object for TeamCRAG
        """

        checkpointer = MemorySaver()

        self.team_app.add_conditional_edges(
            'supervisor', lambda x: x['next'], {'rag': 'rag', 'search': 'search', 'END': END}
        )
        self.team_app.add_edge('rag', 'return_message')
        self.team_app.add_edge('search', 'return_message')
        self.team_app.add_edge('return_message', END)
        self.team_app.set_entry_point('supervisor')
        team_app: CompiledGraph = self.team_app.compile(checkpointer=checkpointer)

        return team_app

    def call_rag(
        self, app: CompiledStateGraph, question: str, config: dict, kwargs: Dict[str, int] = {'recursion_limit': 50}
    ) -> tuple[dict[str, Any], dict[str, Any] | Any]:
        """
        Calls the RAG (Reasoning and Generation) app to generate an answer to a given question.

        Args:
            app: The RAG app object.
            question: The question to be answered.
            kwargs: Keyword arguments to be passed to the app.
            Defaults to {"recursion_limit": 50}
            Recursion limit controls how many runnables to invoke without
            reaching a terminal node.

        Returns:
            response: A dictionary containing the answer and source documents.
                - "answer": The generated answer to the question.
                - "source_documents": A list of source documents used
                to generate the answer.
        """

        response = {}
        # type ignore is used because the invoke expects a runnable,
        # but using a runnable in this way will not work.
        output = app.invoke({'question': question}, config=config)  # type: ignore
        response['answer'] = output['generation']
        sources = output['documents']
        response['source_documents'] = sources

        return response, output
