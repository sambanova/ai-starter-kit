from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM as LangChain_LLM

from financial_agent_crewai.src.tools.general_tools import convert_csv_source_to_txt_report_filename
from financial_agent_crewai.src.tools.rag_tools import TXTSearchTool, TXTSearchToolSchema

load_dotenv()


@CrewBase
class RAGCrew:
    """RAGCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        filename: str,
        llm: LLM,
        rag_llm: LangChain_LLM,
    ) -> None:
        """Initialize the RAGCrew crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.filename = filename
        self.llm = llm
        self.rag_llm = rag_llm

    @agent  # type: ignore
    def rag_researcher(self) -> Agent:
        """Add the RAG Agent."""
        return Agent(
            config=self.agents_config['rag_researcher'],
            verbose=True,
            llm=self.llm,
            tools=[
                TXTSearchTool(
                    txt_path=TXTSearchToolSchema(txt=self.filename),
                    rag_llm=self.rag_llm,
                )
            ],
            allow_delegation=False,
        )

    @task  # type: ignore
    def rag_esearch_task(self) -> Task:
        """Add the RAG Research Task."""
        return Task(
            config=self.tasks_config['rag_research_task'],
            output_file=convert_csv_source_to_txt_report_filename(self.filename),
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the RAGCrew crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
