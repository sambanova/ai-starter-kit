import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

from financial_agent_crewai.src.financial_agent_flow.config import CACHE_DIR, MAX_NEWS
from financial_agent_crewai.utils.utilities import create_log_path

load_dotenv()


@CrewBase
class GenericResearchCrew:
    """GenericResearchCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        cache_dir: Path,
        serper_api_key: str,
        filename: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the GenericResearchCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.llm = llm
        self.cache_dir = cache_dir
        self.filename = filename if filename is not None else str(CACHE_DIR / 'report.txt')
        self.verbose = verbose

        # Set the Google Serper API KEY
        os.environ['SERPER_API_KEY'] = serper_api_key

    @agent  # type: ignore
    def researcher(self) -> Agent:
        """Add the Specialized Finance Researcher Agent."""

        return Agent(
            config=self.agents_config['researcher'],
            verbose=self.verbose,
            llm=self.llm,
            tools=[SerperDevTool(n_results=MAX_NEWS)],
        )

    @task  # type: ignore
    def research_task(self) -> Task:
        """Add the Research Task."""

        return Task(
            config=self.tasks_config['research_task'],
            output_file=self.filename,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the GenericResearchCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
