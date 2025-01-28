from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.general_tools import FilenameOutputList
from financial_agent_crewai.src.tools.yahoo_finance_news import YahooFinanceNewsTool

load_dotenv()


@CrewBase
class YahooFinanceNewsCrew:
    """FinancialAgentCrewai crew."""

    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self, llm: LLM) -> None:
        """Initialize the research crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm

    @agent  # type: ignore
    def yahoo_finance_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['yahoo_finance_researcher'],
            verbose=True,
            llm=self.llm,
            tools=[YahooFinanceNewsTool()],
        )

    @task  # type: ignore
    def yahoo_finance_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['yahoo_finance_research_task'],
            output_pydantic=FilenameOutputList,
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
