import os
import sys
from typing import TypedDict
from langchain_core.prompts import load_prompt
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.graph.graph import CompiledGraph


current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.agents.return_message import ReturnComponents  # type: ignore


class ReturnTeamState(TypedDict):
    question: str
    generation: str
    next: str


class ReturnTeam(ReturnComponents):
    def create_return_team(self) -> CompiledGraph:
        """
        Creates the nodes for the ReturnTeam graph state.

        Args:
            None

        Returns:
            The StateGraph object containing the nodes for the TeamCRAG graph state.
        """

        checkpointer = MemorySaver()

        return_graph: StateGraph = StateGraph(ReturnTeamState)
        return_graph.add_node('return_message', self.return_message_to_user)
        return_graph.add_edge('return_message', END)
        return_graph.set_entry_point('return_message')
        return_graph_compiled: CompiledGraph = return_graph.compile(checkpointer=checkpointer)

        return return_graph_compiled

    def initialize(self):
        """
        Initializes all the components of the ReturnTeam app.

        Args:
            None

        Returns:
            None
        """

        self.init_llm()
        self.init_return_message()
