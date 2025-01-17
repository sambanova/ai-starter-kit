from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from financial_agent_crewai.src.llm import llm
from financial_agent_crewai.src.tools.custom_tools import SecEdgarFilingRetriever, SecEdgarFilingsInput

load_dotenv()


class SECEdgarFilename(BaseModel):
    """SECEdgarFilename model."""

    filename: str = Field(default='', description='The full file path of the SEC Edgar filing.')


@CrewBase
class SECEdgarCrew:
    """SECEdgarCrew crew."""

    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self, input_variables: SecEdgarFilingsInput) -> None:
        """Initialize the research crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.input_variables = input_variables

    @agent  # type: ignore
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['sec_researcher'],
            verbose=True,
            llm=llm,
            tools=[
                SecEdgarFilingRetriever(filing_metadata=self.input_variables),
            ],
        )

    @task  # type: ignore
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            output_pydantic=SECEdgarFilename,
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
