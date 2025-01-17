"""Implementation based on the Crew AI workflow
"""

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from surprise_travel.tools.custom_tool import MyCustomTool

from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List, Optional

import os

# Change the model which you want to use below.
llm = LLM(model="sambanova/Meta-Llama-3.1-70B-Instruct")

@CrewBase
class TravelCrew():
    """Crew to do travel planning"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def personalized_activity_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['personalized_activity_planner'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            allow_delegation=False,
        )

    @agent
    def restaurant_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['restaurant_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @agent
    def museum_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['museum_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @agent
    def shopping_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['shopping_scout'],
            llm=llm,
            max_iter=1,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
        )

    @agent
    def itinerary_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_compiler'],
            llm=llm,
            max_iter=1,
            allow_delegation=False,
        )

    @task
    def personalized_activity_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['personalized_activity_planning_task'],
            llm=llm,
            max_iter=1,
            agent=self.personalized_activity_planner()
        )

    @task
    def restaurant_scenic_location_scout_task(self) -> Task:
        return Task(
            config=self.tasks_config['restaurant_scenic_location_scout_task'],
            llm=llm,
            max_iter=1,
            agent=self.restaurant_scout()
        )

    @task
    def museum_scout_task(self) -> Task:
        return Task(
            config=self.tasks_config['museum_scout_task'],
            llm=llm,
            max_iter=1,
            agent=self.museum_scout()
        )

    @task
    def shopping_scout_task(self) -> Task:
        return Task(
            config=self.tasks_config['shopping_scout_task'],
            llm=llm,
            max_iter=1,
            agent=self.shopping_scout()
        )

    @task
    def itinerary_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config['itinerary_compilation_task'],
            llm=llm,
            max_iter=1,
            agent=self.itinerary_compiler()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Travel crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )


@CrewBase
class AddressSummaryCrew():
    """Address Summary crew"""
    agents_config = 'config/address_agents.yaml'
    tasks_config = 'config/address_tasks.yaml'

    @agent
    def address_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['address_summarizer'],
            llm=llm,
            max_iter=1,
            allow_delegation=False,
        )

    @task
    def address_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config['address_compilation_task'],
            llm=llm,
            max_iter=1,
            agent=self.address_summarizer(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AddressSummary crew"""
        crew = Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
        )
        return crew
