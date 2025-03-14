import datetime
from pathlib import Path
from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import FilenameOutputList
from financial_agent_crewai.src.financial_agent_flow.tools.yfinance_stocks_tools import YFinanceStocksTool
from financial_agent_crewai.utils.utilities import create_log_path

load_dotenv()


@CrewBase
class YFinanceStocksCrew:
    """YFinanceStockCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        query: str,
        ticker_symbol: str,
        llm: LLM,
        pandasai_llm: BaseChatModel,
        start_date: datetime.date,
        end_date: datetime.date,
        cache_dir: Path,
        verbose: bool = True,
    ) -> None:
        """Initialize the YFinanceStockCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.query = query
        self.ticker_symbol = ticker_symbol
        self.llm = llm
        self.pandasai_llm = pandasai_llm
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.verbose = verbose

    @agent  # type: ignore
    def yfinance_stock_analyst(self) -> Agent:
        """Add the Yahoo Finance Stock Analyst Agent."""

        return Agent(
            config=self.agents_config['yfinance_stock_analyst'],
            verbose=self.verbose,
            llm=self.llm,
            task='yfinance_stock_analysis',
            tools=[
                YFinanceStocksTool(
                    llm=self.pandasai_llm,
                    query=self.query,
                    ticker_symbol=self.ticker_symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    cache_dir=self.cache_dir,
                ),
            ],
        )

    @task  # type: ignore
    def yfinance_stock_analysis(self) -> Task:
        """Add the YFinance Stock Analysis Task."""

        return Task(
            config=self.tasks_config['yfinance_stock_analysis'],
            output_pydantic=FilenameOutputList,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the YFinanceStockCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
