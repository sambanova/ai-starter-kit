from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.general_tools import FilenameOutput
from financial_agent_crewai.src.tools.sec_edgar_tools import SecEdgarFilingsInput

load_dotenv()


@CrewBase
class YFinanceStockCrew:
    """YFinanceStockCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(self, input_variables: SecEdgarFilingsInput, llm: LLM) -> None:
        """Initialize the YFinanceStockCrew crew."""

        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.input_variables = input_variables
        self.llm = llm

    @agent  # type: ignore
    def yfinance_stock_analyst(self) -> Agent:
        """Add the Yahoo Finance Stock Analyst Agent."""
        return Agent(
            config=self.agents_config['yfinance_stock_analyst'],
            verbose=True,
            llm=self.llm,
            task='yfinance_stock_analysis',
            # tools=[
            #     SecEdgarFilingRetriever(filing_metadata=self.input_variables),
            # ],
        )

    @task  # type: ignore
    def yfinance_stock_analysis(self) -> Task:
        """Add the YFinance Stock Analysis Task."""
        return Task(
            config=self.tasks_config['yfinance_stock_analysis'],
            output_pydantic=FilenameOutput,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the YFinanceStockCrew crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
