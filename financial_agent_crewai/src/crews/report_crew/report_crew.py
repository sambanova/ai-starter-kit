from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.report_tools import ReportSection

load_dotenv()


@CrewBase
class ReportCrew:
    """ReportCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        verbose: bool = True,
    ) -> None:
        """Initialize the ReportCrew crew."""

        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm
        self.verbose = verbose

    @agent  # type: ignore
    def reporting_analyst(self) -> Agent:
        """Add the Finance Reporting Analyst Agent."""

        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=self.verbose,
            llm=self.llm,
        )

    @task  # type: ignore
    def reporting_task(self) -> Task:
        """Add the Reporting Task."""

        return Task(
            config=self.tasks_config['reporting_task'],
            output_pydantic=ReportSection,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the ReportCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )
