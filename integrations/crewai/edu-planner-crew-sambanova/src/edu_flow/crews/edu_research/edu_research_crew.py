import os
import sys

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

current_dir = os.getcwd()
repo_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(repo_dir)

from typing import List

# Uncomment the following line to use an example of a custom tool
# from edu_research.tools.custom_tool import MyCustomTool
# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool
from pydantic import BaseModel

# Define the LLM instance at the top level
from src.edu_flow.llm_config import llm


class Section(BaseModel):
    title: str
    high_level_goal: str
    why_important: str
    sources: List[str]
    content_outline: List[str]


class EducationalPlan(BaseModel):
    sections: List[Section]


@CrewBase
class EduResearchCrew:
    """EduResearch crew"""

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'], llm=llm, verbose=True, tools=[SerperDevTool()])

    @agent
    def planner(self) -> Agent:
        return Agent(config=self.agents_config['planner'], llm=llm, verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config['planning_task'], output_pydantic=EducationalPlan)

    @crew
    def crew(self) -> Crew:
        """Creates the EduResearch crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
