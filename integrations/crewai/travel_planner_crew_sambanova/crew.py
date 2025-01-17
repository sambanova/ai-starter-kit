"""Implementation based on the Crew AI workflow"""

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Change the model which you want to use below.
# Currently, we use the Llama 3.1 70B model because it seems the most versatile
llm = LLM(model='sambanova/Meta-Llama-3.1-70B-Instruct')

@CrewBase
class TravelCrew:
    """Crew to do travel planning"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

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

    @agent
    def museum_scout(self) -> Agent:
        """
        An agent specialized to scout for museums

        Returns: The agent
        """
        return Agent(
            config=self.agents_config['museum_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @agent
    def shopping_scout(self) -> Agent:
        """
        An agent specialized to scout for sopping options

        Returns: The agent
        """

        return Agent(
            config=self.agents_config['shopping_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

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

    @task
    def museum_scout_task(self) -> Task:
        """
        A task that plans for museums.

        Returns: A task
        """
        return Task(config=self.tasks_config['museum_scout_task'], llm=llm, max_iter=1, agent=self.museum_scout())

    @task
    def shopping_scout_task(self) -> Task:
        """
        A task that plans for shopping.

        Returns: A task
        """
        return Task(config=self.tasks_config['shopping_scout_task'], llm=llm, max_iter=1, agent=self.shopping_scout())

    @task
    def itinerary_compilation_task(self) -> Task:
        """
        A task that plans for museums.

        Returns: A task
        """
        return Task(
            config=self.tasks_config['itinerary_compilation_task'], llm=llm, max_iter=1, agent=self.itinerary_compiler()
        )

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
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )


@CrewBase
class AddressSummaryCrew:
    """Address Summary crew"""

    agents_config = 'config/address_agents.yaml'
    tasks_config = 'config/address_tasks.yaml'

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
