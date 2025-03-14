from pathlib import Path
from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import FilenameOutput
from financial_agent_crewai.src.financial_agent_flow.tools.sec_edgar_tools import SecEdgarFilingRetriever
from financial_agent_crewai.src.financial_agent_flow.tools.sorting_hat_tools import FilingsInput
from financial_agent_crewai.utils.utilities import create_log_path

load_dotenv()


@CrewBase
class SECEdgarCrew:
    """SECEdgarCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        input_variables: FilingsInput,
        llm: LLM,
        cache_dir: Path,
        verbose: bool = True,
    ) -> None:
        """Initialize the SECEdgarCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.input_variables = input_variables
        self.llm = llm
        self.cache_dir = cache_dir
        self.verbose = verbose

    @agent  # type: ignore
    def sec_researcher(self) -> Agent:
        """Add the SEC EDGAR Curator Agent."""

        return Agent(
            config=self.agents_config['sec_researcher'],
            verbose=self.verbose,
            llm=self.llm,
            tools=[SecEdgarFilingRetriever(filing_metadata=self.input_variables, cache_dir=self.cache_dir)],
        )

    @task  # type: ignore
    def sec_research_task(self) -> Task:
        """Add the SEC Research Task."""

        return Task(
            config=self.tasks_config['sec_research_task'],
            output_pydantic=FilenameOutput,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the SECEdgarCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
