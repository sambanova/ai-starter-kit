from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.general_tools import FilenameOutput
from financial_agent_crewai.src.tools.yahoo_finance_news_tools import YahooFinanceNewsTool

load_dotenv()


@CrewBase
class YahooFinanceNewsCrew:
    """YahooFinanceNewsCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(self, llm: LLM) -> None:
        """Initialize the YahooFinanceNewsCrew crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm

    @agent  # type: ignore
    def yahoo_finance_researcher(self) -> Agent:
        """Add the Yahoo Finance News Curator Agent."""
        return Agent(
            config=self.agents_config['yahoo_finance_researcher'],
            verbose=True,
            llm=self.llm,
            tools=[YahooFinanceNewsTool()],
        )

    @task  # type: ignore
    def yahoo_finance_research_task(self) -> Task:
        """Add the Yahoo Finance Research Task."""
        return Task(
            config=self.tasks_config['yahoo_finance_research_task'],
            output_pydantic=FilenameOutput,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the YahooFinanceNewsCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
