from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
# Uncomment the following line to use an example of a custom tool
# from latest_ai_development.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
from financial_assistant.src.tools_filings import retrieve_filings
from crewai import LLM
# # # Set Up Sambanova

# from utils import get_sambanova_api_key, update_task_output_format

llm = LLM(
    model='sambanova/Meta-Llama-3.1-70B-Instruct',
    api_key=os.getenv('SAMBANOVA_API_KEY'),
    base_url=os.getenv('SAMBANOVA_BASE_URL'),
)
manager_llm = LLM(
    model='sambanova/Meta-Llama-3.1-70B-Instruct',
    api_key=os.getenv('SAMBANOVA_API_KEY'),
    base_url=os.getenv('SAMBANOVA_BASE_URL'),
)


@CrewBase
class LatestAiDevelopment:
    """LatestAiDevelopment crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    from crewai_tools import tool, Tool

    tool_filings = Tool(
        name='retrieve_filings',
        description=retrieve_filings.description,
        func=retrieve_filings,
        pydantic_output=retrieve_filings.args_schema,
    )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            llm=llm,
            tools=[self.tool_filings],  # Example of custom tool, loaded on the beginning of file
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            llm=llm,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            tools=[self.tool_filings],
            pydantic_output=retrieve_filings.args_schema,
        )

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config['reporting_task'], output_file='report.md')

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
