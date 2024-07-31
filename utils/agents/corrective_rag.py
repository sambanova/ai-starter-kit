from typing import Any, Dict, List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.config import RunnableConfig
from utils.rag.rag_components import RAGComponents  # type: ignore
from utils.code_gen.codegen_components import CodeGenComponents  # type: ignore


class RAGGraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    answers: List[str]
    original_question: str
    """
    Represents the state of a RAG (Retrieval Augmented Generation) graph.

    Args:
        question: The question being asked by the user.
        generation: The most recent generated text from the LLM agent.
        documents: A list of relevant documents retrieved from the vectorstore.
        answers: A list of possible answers that have been accumulated by the LLM agents.
        original_question: The original question asked - in case of subquery generation, etc..
    """


class CorrectiveRAG(RAGComponents, CodeGenComponents):
    def create_rag_nodes(self) -> StateGraph:
        """
        Creates the nodes for the CorrectiveRAG graph state.

        Args:
            None

        Returns:
            The StateGraph object containing the nodes for the CodeRAG graph state.
        """

        workflow: StateGraph = StateGraph(RAGGraphState)

        # Define the nodes

        workflow.add_node('initialize', self.initialize_rag)
        workflow.add_node('retrieve', self.retrieve)
        workflow.add_node('grade_documents', self.grade_documents)
        workflow.add_node('generate', self.rag_generate)
        workflow.add_node('failure_msg', self.failure_msg)
        workflow.add_node('return_final_answer', self.final_answer)

        return workflow

    def build_rag_graph(self, workflow: StateGraph) -> CompiledGraph:
        """
        Builds a graph for the RAG workflow.

        This method constructs a workflow graph that represents the sequence of tasks
        performed by the RAG system. The graph is used to execute the workflow and
        generate code.

        Args:
            workflow: The workflow object (StateGraph containing nodes) to be modified.

        Returns:
            The compiled application object for static CodeRAG
        """

        checkpointer = MemorySaver()

        workflow.set_entry_point('initialize')
        workflow.add_edge('initialize', 'retrieve')
        workflow.add_edge('retrieve', 'grade_documents')
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
        Initializes all the components of the static CorrectiveRAG app.
        """

        self.init_llm()
        self.init_retrieval_grader()
        self.init_qa_chain()
        self.init_retrieval_grader()
        self.init_hallucination_chain()
        self.init_grading_chain()
        self.init_failure_chain()
        self.init_final_generation()

    def call_rag(
        self, app: CompiledStateGraph, question: str, kwargs: Dict[str, int] = {'recursion_limit': 50}
    ) -> tuple[dict[str, Any], Dict[str, Any] | Any]:
        """
        Calls the RAG (Reasoning and Generation) app to generate an answer to a given question.

        Args:
            app: The RAG app object.
            question: The question to be answered.
            kwargs: Keyword arguments to be passed to the app.
            Defaults to {"recursion_limit": 50}.
            Recursion limit controls how many runnables to invoke without
            reaching a terminal node.

        Returns:
            response: A dictionary containing the answer and source documents.
                - "answer" (str): The generated answer to the question.
                - "source_documents" (List[str]): A list of source documents used
                to generate the answer.
        """

        runnable = RunnableConfig(configurable={'question': question})

        response = {}
        output = app.invoke(runnable, kwargs=kwargs)
        response['answer'] = output['generation']
        sources = output['documents']
        response['source_documents'] = sources

        return response, output
