from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.sec_edgar_tools import SecEdgarFilingsInputsList

load_dotenv()
from crewai import LLM


@CrewBase
class SortingHatCrew:
    """SortingHatCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(self, llm: LLM) -> None:
        """Initialize the SortingHatCrew crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm

    @agent  # type: ignore
    def extractor(self) -> Agent:
        """Add the Information Extractor Agent."""
        return Agent(
            config=self.agents_config['extractor'],
            verbose=True,
            llm=self.llm,
        )

    @task  # type: ignore
    def extraction_task(self) -> Task:
        """Add the Extraction Task."""
        return Task(
            config=self.tasks_config['extraction_task'],
            output_pydantic=SecEdgarFilingsInputsList,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the SortingHatCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
