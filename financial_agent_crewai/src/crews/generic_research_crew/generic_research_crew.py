from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

from financial_agent_crewai.src.utils.config import CACHE_DIR

load_dotenv()


@CrewBase
class GenericResearchCrew:
    """GenericResearchCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        filename: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the GenericResearchCrew crew."""

        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm
        self.filename = filename if filename is not None else str(CACHE_DIR / 'report.txt')
        self.verbose = verbose

    @agent  # type: ignore
    def researcher(self) -> Agent:
        """Add the Specialized Finance Researcher Agent."""

        return Agent(
            config=self.agents_config['researcher'],
            verbose=self.verbose,
            llm=self.llm,
            tools=[SerperDevTool()],
        )

    @task  # type: ignore
    def research_task(self) -> Task:
        """Add the Research Task."""

        return Task(
            config=self.tasks_config['research_task'],
            output_file=self.filename,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the GenericResearchCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )
