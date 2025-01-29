from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

from financial_agent_crewai.src.utils.config import CACHE_DIR

load_dotenv()


@CrewBase
class GenericResearchCrew:
    """FinancialAgentCrewai crew."""

    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self, llm: LLM, filename: Optional[str] = None) -> None:
        """Initialize the research crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm
        self.filename = filename if filename is not None else str(CACHE_DIR / 'report.txt')

    @agent  # type: ignore
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            llm=self.llm,
            tools=[SerperDevTool()],
        )

    @task  # type: ignore
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            output_file=self.filename,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Creates the FinancialAgentCrewai crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
