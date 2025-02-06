from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.general_tools import SubQueriesList

load_dotenv()


@CrewBase
class DecompositionCrew:
    """DecompositionCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        verbose: bool = True,
    ) -> None:
        """Initialize the DecompositionCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.llm = llm
        self.verbose = verbose

    @agent  # type: ignore
    def reformulator(self) -> Agent:
        """Add the Reformulator Agent."""

        return Agent(
            config=self.agents_config['reformulator'],
            verbose=self.verbose,
            llm=self.llm,
        )

    @task  # type: ignore
    def reformulation_task(self) -> Task:
        """Add the Reformulation Task."""

        return Task(
            config=self.tasks_config['reformulation_task'],
            output_pydantic=SubQueriesList,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the DecompositionCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )
