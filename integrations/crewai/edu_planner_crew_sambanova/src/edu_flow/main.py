#!/usr/bin/env python
"""
Main module for the Educational Content Generation Flow.

This module implements a workflow for generating educational content using
multiple specialized AI crews. It handles the coordination between research
and content creation phases.
"""

import os
from typing import List

from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv

from .config import EDU_FLOW_INPUT_VARIABLES
from .crews.edu_content_writer.edu_content_writer_crew import EduContentWriterCrew
from .crews.edu_research.edu_research_crew import EducationalPlan, EduResearchCrew

load_dotenv()


class EduFlow(Flow):
    """
    Educational content generation workflow manager.

    This class orchestrates the process of researching topics and generating
    educational content through multiple specialized AI crews.

    Attributes:
        input_variables (dict): Configuration for the educational content generation
        research_crew (Crew): Crew responsible for research phase
        content_crew (Crew): Crew responsible for content creation phase
    """

    input_variables = EDU_FLOW_INPUT_VARIABLES

    def __init__(self) -> None:
        """Initialize the educational flow with research and content creation crews."""
        super().__init__()
        self.research_crew = EduResearchCrew().crew()
        self.content_crew = EduContentWriterCrew().crew()

    @start()
    def generate_reseached_content(self) -> EducationalPlan:
        """
        Begin the content generation process with research.

        Returns:
            EducationalPlan: A structured plan for educational content based on research.
        """
        return self.research_crew.kickoff(self.input_variables).pydantic

    @listen(generate_reseached_content)
    def generate_educational_content(self, plan: EducationalPlan) -> List[str]:
        """
        Generate educational content based on the research plan.

        Args:
            plan (EducationalPlan): The structured content plan from research phase.

        Returns:
            List[str]: List of generated content sections.
        """
        final_content: List[str] = []

        for section in plan.sections:
            writer_inputs = self.input_variables.copy()
            writer_inputs['section'] = section.model_dump_json()
            final_content.append(self.content_crew.kickoff(writer_inputs).raw)

        return final_content

    @listen(generate_educational_content)
    def save_to_markdown(self, content: List[str]) -> None:
        """
        Save the generated content to a markdown file.

        Args:
            content (List[str]): List of content sections to save.
        """
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        topic = self.input_variables.get('topic')
        audience_level = self.input_variables.get('audience_level')
        file_name = f'{topic}_{audience_level}.md'.replace(' ', '_')

        output_path = os.path.join(output_dir, file_name)

        with open(output_path, 'w') as f:
            for section in content:
                f.write(section)
                f.write('\n\n')  # Add space between sections


def kickoff() -> None:
    """Initialize and start the educational content generation process."""
    edu_flow = EduFlow()
    edu_flow.kickoff()


def plot() -> None:
    """Generate and display a visualization of the flow structure."""
    edu_flow = EduFlow()
    edu_flow.plot()


if __name__ == '__main__':
    kickoff()
