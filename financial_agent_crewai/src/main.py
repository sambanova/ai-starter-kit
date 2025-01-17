#!/usr/bin/env python
import warnings
from typing import Any, List

from financial_agent_crewai.src.crews.decomposition_crew.decomposition_crew import (
    DecompositionCrew,
)
from financial_agent_crewai.src.crews.generic_research_crew.generic_research_crew import GenericResearchCrew
from financial_agent_crewai.src.crews.rag_crew.rag_crew import RAGCrew
from financial_agent_crewai.src.tools.custom_tools import SecEdgarFilingsInputsList

warnings.filterwarnings('ignore', category=SyntaxWarning, module='pysbd')

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

#!/usr/bin/env python
"""
Main module for the Educational Content Generation Flow.

This module implements a workflow for generating educational content using
multiple specialized AI crews. It handles the coordination between research
and content creation phases.
"""


from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv

from financial_agent_crewai.src.crews.decomposition_crew.decomposition_crew import DecompositionCrew

# from .config import EDU_FLOW_INPUT_VARIABLES
# from .crews.edu_content_writer.edu_content_writer_crew import EduContentWriterCrew
# from .crews.edu_research.edu_research_crew import EducationalPlan, EduResearchCrew
from financial_agent_crewai.src.crews.generic_research_crew.generic_research_crew import GenericResearchCrew
from financial_agent_crewai.src.crews.sec_edgar_crew.sec_edgar_crew import SECEdgarCrew

load_dotenv()


class FinancialFlow(Flow):  # type: ignore
    """
    Educational content generation workflow manager.

    This class orchestrates the process of researching topics and generating
    educational content through multiple specialized AI crews.

    Attributes:
        input_variables (dict): Configuration for the educational content generation
        research_crew (Crew): Crew responsible for research phase
        content_crew (Crew): Crew responsible for content creation phase
    """

    # input_variables = EDU_FLOW_INPUT_VARIABLES

    def __init__(self) -> None:
        """Initialize the educational flow with research and content creation crews."""
        super().__init__()
        self.research_crew = GenericResearchCrew().crew()
        # self.content_crew = EduContentWriterCrew().crew()

    @start()  # type: ignore
    def extract_information(self) -> Any:
        """
        Begin the content generation process with research.

        Returns:
            EducationalPlan: A structured plan for educational content based on research.
        """

        return (
            DecompositionCrew()
            .crew()
            .kickoff(inputs={'query': 'What are the conclusions of the latest 10-K filings for Meta in 2023?'})
            .pydantic.inputs_list
        )

    @listen(extract_information)  # type: ignore
    def sec_edgar_research(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> Any:
        """xxx"""

        sec_edgar_filenames_list = list()
        for filing_metadata in sec_edgar_inputs_list:
            sec_edgar_filenames_list.append(
                SECEdgarCrew(input_variables=filing_metadata)  # type: ignore
                .crew()
                .kickoff(
                    {
                        'query': 'What are the conclusions of the latest 10-K filings for Meta in 2023?',
                    },
                )
                .pydantic.filename
            )
        return sec_edgar_filenames_list

    @listen(sec_edgar_research)  # type: ignore
    def sec_edgar_rag(self, sec_edgar_filenames_list: List[str]) -> Any:
        """xxx"""

        sec_reports_list = list()
        for filename in sec_edgar_filenames_list:
            sec_reports_list.append(
                RAGCrew(filename=filename)
                .crew()
                .kickoff(
                    {
                        'query': 'What are the conclusions of the latest 10-K filings for Meta in 2023?',
                    },
                )
            )

        return sec_reports_list

    # @listen(generate_reseached_content)  # type: ignore
    # def generate_educational_content(self, plan: str) -> List[str]:
    #     """
    #     Generate educational content based on the research plan.

    #     Args:
    #         plan (EducationalPlan): The structured content plan from research phase.

    #     Returns:
    #         List[str]: List of generated content sections.
    #     """
    #     final_content: List[str] = []

    #     for section in plan.sections:  # type: ignore
    #         writer_inputs = self.input_variables.copy()
    #         writer_inputs['section'] = section.model_dump_json()
    #         final_content.append(self.content_crew.kickoff(writer_inputs).raw)

    #     return final_content

    # @listen(generate_educational_content)  # type: ignore
    # def save_to_markdown(self, content: List[str]) -> None:
    #     """
    #     Save the generated content to a markdown file.

    #     Args:
    #         content (List[str]): List of content sections to save.
    #     """
    #     output_dir = 'output'
    #     os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    #     topic = self.input_variables.get('topic')
    #     audience_level = self.input_variables.get('audience_level')
    #     file_name = f'{topic}_{audience_level}.md'.replace(' ', '_')

    #     output_path = os.path.join(output_dir, file_name)

    #     with open(output_path, 'w') as f:
    #         for section in content:
    #             f.write(section)
    #             f.write('\n\n')  # Add space between sections


def kickoff() -> None:
    """Initialize and start the educational content generation process."""
    edu_flow = FinancialFlow()
    edu_flow.kickoff()


def plot() -> None:
    """Generate and display a visualization of the flow structure."""
    edu_flow = FinancialFlow()
    edu_flow.plot()


if __name__ == '__main__':
    plot()
    kickoff()


# def run() -> None:
#     """
#     Run the crew.
#     """
#     inputs = {'topic': 'AI LLMs'}
#     FinancialAgentCrew().crew().kickoff(inputs=inputs)


# def train() -> None:
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {'topic': 'AI LLMs'}
#     try:
#         FinancialAgentCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f'An error occurred while training the crew: {e}')


# def replay() -> None:
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         FinancialAgentCrew().crew().replay(task_id=sys.argv[1])

#     except Exception as e:
#         raise Exception(f'An error occurred while replaying the crew: {e}')


# def test() -> None:
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {'topic': 'AI LLMs'}
#     try:
#         FinancialAgentCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f'An error occurred while replaying the crew: {e}')
