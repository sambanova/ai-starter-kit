from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.utils.utilities import create_log_path

load_dotenv()


@CrewBase
class ContextAnalysisCrew:
    """ContextAnalysisCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(
        self,
        llm: LLM,
        cache_dir: Path,
        output_file: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the ContextAnalysisCrew crew."""

        super().__init__()
        self.agents_config = dict()
        self.tasks_config = dict()
        self.agents = list()
        self.tasks = list()
        self.llm = llm
        self.cache_dir = cache_dir
        self.output_file = output_file if output_file else 'context_analysis.txt'
        self.verbose = verbose

    @agent  # type: ignore
    def context_analyst(self) -> Agent:
        """Add the Finance Reporting Analyst Agent."""

        return Agent(
            config=self.agents_config['context_analyst'],
            verbose=self.verbose,
            llm=self.llm,
        )

    @task  # type: ignore
    def context_analysis_task(self) -> Task:
        """Add the Context Analysis Task."""

        return Task(
            config=self.tasks_config['context_analysis_task'],
            output_file=self.output_file,
        )

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Create the ContextAnalysisCrew crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            output_log_file=create_log_path(self.cache_dir),
        )
