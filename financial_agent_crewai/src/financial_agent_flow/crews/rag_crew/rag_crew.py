from pathlib import Path
from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import (
    convert_csv_source_to_txt_report_filename,
)
from financial_agent_crewai.src.financial_agent_flow.tools.rag_tools import TXTSearchTool, TXTSearchToolSchema
from financial_agent_crewai.utils.utilities import create_log_path

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
        cache_dir: Path,
        verbose: bool = True,
    ) -> None:
        """Initialize the RAGCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.filename = filename
        self.llm = llm
        self.cache_dir = cache_dir
        self.verbose = verbose

    @agent  # type: ignore
    def rag_researcher(self) -> Agent:
        """Add the RAG Agent."""

        return Agent(
            config=self.agents_config['rag_researcher'],
            verbose=self.verbose,
            llm=self.llm,
            tools=[
                TXTSearchTool(
                    txt_path=TXTSearchToolSchema(txt=self.filename),
                    rag_llm=self.llm,
                )
            ],
            allow_delegation=False,
        )

    @task  # type: ignore
    def rag_research_task(self) -> Task:
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
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
