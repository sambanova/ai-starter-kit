from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()


@CrewBase
class ContextAnalysisCrew:
    """FinancialAgentCrewai crew."""

    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self, llm: LLM, output_file: Optional[str] = None) -> None:
        """Initialize the research crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.llm = llm
        self.output_file = output_file if output_file else 'context_analysis.txt'

    @agent  # type: ignore
    def context_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['context_analyst'],
            verbose=True,
            llm=self.llm,
        )

    @task  # type: ignore
    def context_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['context_analysis_task'], output_file=self.output_file)

    @crew  # type: ignore
    def crew(self) -> Crew:
        """Creates the FinancialAgentCrewai crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
