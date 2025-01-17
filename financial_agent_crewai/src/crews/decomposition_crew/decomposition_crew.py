from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
from dotenv import load_dotenv

from financial_agent_crewai.src.llm import llm
from financial_agent_crewai.src.tools.custom_tools import SecEdgarFilingsInputsList

# Set up your SERPER_API_KEY key in an .env file, eg:
# SERPER_API_KEY=<your api key>
load_dotenv()


@CrewBase
class DecompositionCrew:
    """SECEdgarCrew crew."""

    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self) -> None:
        """Initialize the research crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []

    @agent  # type: ignore
    def extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['extractor'],
            verbose=True,
            llm=llm,
            task='extraction_task',
            memory=True,
        )

    @task  # type: ignore
    def extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['extraction_task'],
            output_pydantic=SecEdgarFilingsInputsList,
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
