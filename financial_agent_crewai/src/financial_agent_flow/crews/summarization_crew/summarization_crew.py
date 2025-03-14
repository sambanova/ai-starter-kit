from pathlib import Path
from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.financial_agent_flow.tools.report_tools import ReportSummary
from financial_agent_crewai.utils.utilities import create_log_path

load_dotenv()


@CrewBase
class SummarizationCrew:
    """ReportCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        cache_dir: Path,
        verbose: bool = True,
    ) -> None:
        """Initialize the ReportCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.llm = llm
        self.cache_dir = cache_dir
        self.verbose = verbose

    @agent  # type: ignore
    def summarizer(self) -> Agent:
        """Add the Finance Reporting Analyst Agent."""

        return Agent(
            config=self.agents_config['summarizer'],
            verbose=self.verbose,
            llm=self.llm,
        )

    @task  # type: ignore
    def summarization(self) -> Task:
        """Add the Reporting Task."""

        return Task(
            config=self.tasks_config['summarization'],
            output_pydantic=ReportSummary,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the ReportCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
