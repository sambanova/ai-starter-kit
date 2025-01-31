from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from financial_agent_crewai.src.tools.general_tools import FilenameOutput
from financial_agent_crewai.src.tools.sec_edgar_tools import SecEdgarFilingRetriever, SecEdgarFilingsInput

load_dotenv()


@CrewBase
class SECEdgarCrew:
    """SECEdgarCrew crew."""

    agents_config: Dict[str, Any]
    tasks_config: Dict[str, Any]
    agents: List[Any]
    tasks: List[Any]

    def __init__(self, input_variables: SecEdgarFilingsInput, llm: LLM) -> None:
        """Initialize the SECEdgarCrew crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.input_variables = input_variables
        self.llm = llm

    @agent  # type: ignore
    def sec_researcher(self) -> Agent:
        """Add the SEC EDGAR Curator Agent."""
        return Agent(
            config=self.agents_config['sec_researcher'],
            verbose=True,
            llm=self.llm,
            tools=[SecEdgarFilingRetriever(filing_metadata=self.input_variables)],
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
            verbose=True,
        )
