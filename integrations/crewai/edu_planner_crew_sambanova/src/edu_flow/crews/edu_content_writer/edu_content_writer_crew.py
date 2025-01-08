"""
Module for handling educational content generation through a crew of specialized agents.

This module implements a CrewAI-based content generation system with multiple agents
working together to create, edit, and review educational content.
"""

import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.edu_flow.config import EDU_FLOW_INPUT_VARIABLES
from src.edu_flow.llm_config import llm


@CrewBase
class EduContentWriterCrew:
    """
    A crew of AI agents specialized in creating educational content.

    This crew consists of three main agents:
    - Content Writer: Creates initial educational content
    - Editor: Refines and improves the content
    - Quality Reviewer: Ensures content meets educational standards

    Attributes:
        input_variables (dict): Configuration variables for educational content generation
    """

    input_variables = EDU_FLOW_INPUT_VARIABLES
    agents_config: Dict[str, Any]  # Type hint for the config attribute
    tasks_config: Dict[str, Any]  # Type hint for the tasks config
    agents: List[Any]  # Type hint for the agents list
    tasks: List[Any]  # Type hint for the tasks list

    def __init__(self) -> None:
        """Initialize the content writer crew."""
        super().__init__()
        self.agents_config = {}
        self.tasks_config = {}
        self.agents = []
        self.tasks = []
        self.__post_init__()

    def __post_init__(self) -> None:
        """Initialize the crew by ensuring required directories exist."""
        self.ensure_output_folder_exists()

    def ensure_output_folder_exists(self) -> None:
        """
        Create the output directory if it doesn't exist.

        This method ensures that the 'output' directory is available for storing
        generated content files.
        """
        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @agent
    def content_writer(self) -> Agent:
        """
        Create the content writer agent.

        Returns:
            Agent: An AI agent specialized in creating educational content.
        """
        return Agent(config=self.agents_config['content_writer'], llm=llm, verbose=True)

    @agent
    def editor(self) -> Agent:
        """
        Create the editor agent.

        Returns:
            Agent: An AI agent specialized in editing and refining content.
        """
        return Agent(config=self.agents_config['editor'], llm=llm, verbose=True)

    @agent
    def quality_reviewer(self) -> Agent:
        """
        Create the quality reviewer agent.

        Returns:
            Agent: An AI agent specialized in reviewing and ensuring content quality.
        """
        return Agent(config=self.agents_config['quality_reviewer'], llm=llm, verbose=True)

    @task
    def writing_task(self) -> Task:
        """
        Define the initial content writing task.

        Returns:
            Task: A task configuration for content creation.
        """
        return Task(
            config=self.tasks_config['writing_task'],
        )

    @task
    def editing_task(self) -> Task:
        """
        Define the content editing task.

        This task includes file path configuration based on the topic and audience level.

        Returns:
            Task: A task configuration for content editing.
        """
        topic = self.input_variables.get('topic')
        audience_level = self.input_variables.get('audience_level')
        file_name = f'{topic}_{audience_level}.md'.replace(' ', '_')
        output_file_path = os.path.join('output', file_name)

        return Task(config=self.tasks_config['editing_task'], output_file=output_file_path)

    @task
    def quality_review_task(self) -> Task:
        """
        Define the quality review task.

        Returns:
            Task: A task configuration for quality review.
        """
        return Task(
            config=self.tasks_config['quality_review_task'],
        )

    @crew
    def crew(self) -> Crew:
        """
        Create and configure the content creation crew.

        Returns:
            Crew: A configured crew with all necessary agents and tasks.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
