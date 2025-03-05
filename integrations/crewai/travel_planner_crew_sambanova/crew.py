"""Implementation based on the Crew AI workflow"""

import typing

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Import tools from crewai_tools
# ScrapWebsiteTool: Extract content of wensite, https://docs.crewai.com/tools/scrapewebsitetool
# SerperDevTool: Search Internet, https://docs.crewai.com/tools/serperdevtool
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Change the model which you want to use below.
# Currently, we use the Llama 3.1 70B model because it seems the most versatile.
llm = LLM(model='sambanova/Meta-Llama-3.1-70B-Instruct')


@CrewBase
class TravelCrew:
    """Crew to do travel planning"""

    def __init__(self) -> None:
        """Initialize the crew."""
        super().__init__()
        self.agents_config = 'config/agents.yaml'
        self.tasks_config = 'config/tasks.yaml'

    @typing.no_type_check
    @agent
    def personalized_activity_planner(self) -> Agent:
        """
        An agent specialized to build an activity planner

        Returns: The agent
        """
        return Agent(
            config=self.agents_config['personalized_activity_planner'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],  # Example of custom tool, loaded at the beginning of file
            allow_delegation=False,
        )

    @typing.no_type_check
    @agent
    def restaurant_scout(self) -> Agent:
        """
        An agent specialized to scout for restaurants

        Returns: The agent
        """
        return Agent(
            config=self.agents_config['restaurant_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @typing.no_type_check
    @agent
    def interest_scout(self) -> Agent:
        """
        An agent specialized to scout for specific interests

        Returns: The agent
        """
        return Agent(
            config=self.agents_config['interest_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @typing.no_type_check
    @agent
    def itinerary_compiler(self) -> Agent:
        """
        An agent specialized at composing the entire itinirary

        Returns: The agent
        """
        return Agent(
            config=self.agents_config['itinerary_compiler'],
            llm=llm,
            max_iter=1,
            allow_delegation=False,
        )

    @typing.no_type_check
    @task
    def personalized_activity_planning_task(self) -> Task:
        """
        A task that designs and plans for activities.

        Returns: A task
        """
        return Task(
            config=self.tasks_config['personalized_activity_planning_task'],
            llm=llm,
            max_iter=1,
            agent=self.personalized_activity_planner(),
        )

    @typing.no_type_check
    @task
    def interest_scout_task(self) -> Task:
        """
        A task that plans for specific interests of the traveller.

        Returns: A task
        """
        return Task(config=self.tasks_config['interest_scout_task'], llm=llm, max_iter=1, agent=self.interest_scout())

    @typing.no_type_check
    @task
    def restaurant_scenic_location_scout_task(self) -> Task:
        """
        A task that picks restaurants.

        Returns: A task
        """
        return Task(
            config=self.tasks_config['restaurant_scenic_location_scout_task'],
            llm=llm,
            max_iter=1,
            agent=self.restaurant_scout(),
        )

    @typing.no_type_check
    @task
    def itinerary_compilation_task(self) -> Task:
        """
        A task that plans for museums.

        Returns: A task
        """
        return Task(
            config=self.tasks_config['itinerary_compilation_task'], llm=llm, max_iter=1, agent=self.itinerary_compiler()
        )

    @typing.no_type_check
    @crew
    def crew(self) -> Crew:
        """
        Creates the Travel Planning crew

        Returns: A crew
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )


@CrewBase
class AddressSummaryCrew:
    """Address Summary crew"""

    agents_config = 'config/address_agents.yaml'
    tasks_config = 'config/address_tasks.yaml'

    @typing.no_type_check
    @agent
    def address_summarizer(self) -> Agent:
        """
        Creates an agent which can summarize addresses in a Json file

        Returns: An agent
        """
        return Agent(
            config=self.agents_config['address_summarizer'],
            llm=llm,
            max_iter=1,
            allow_delegation=False,
        )

    @typing.no_type_check
    @task
    def address_compilation_task(self) -> Task:
        """
        Creates a task which can summarize addresses

        Returns: A Task
        """
        return Task(
            config=self.tasks_config['address_compilation_task'],
            llm=llm,
            max_iter=1,
            agent=self.address_summarizer(),
        )

    @typing.no_type_check
    @crew
    def crew(self) -> Crew:
        """
        Creates the AddressSummary crew

        Returns: A Crew
        """
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )
        return crew
