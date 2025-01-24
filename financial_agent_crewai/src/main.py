"""
Main module for the Educational Content Generation Flow.

This module implements a workflow for generating educational content using
multiple specialized AI crews. It handles the coordination between research
and content creation phases.
"""

import logging
import os
import warnings
from typing import Any, List

from crewai.flow.flow import Flow, and_, listen, start
from dotenv import load_dotenv

from financial_agent_crewai.src.crews.decomposition_crew.decomposition_crew import (
    DecompositionCrew,
)

# from .config import EDU_FLOW_INPUT_VARIABLES
# from .crews.edu_content_writer.edu_content_writer_crew import EduContentWriterCrew
# from .crews.edu_research.edu_research_crew import EducationalPlan, EduResearchCrew
from financial_agent_crewai.src.crews.generic_research_crew.generic_research_crew import GenericResearchCrew
from financial_agent_crewai.src.crews.information_extraction_crew.information_extraction_crew import (
    InformationExtractionCrew,
)
from financial_agent_crewai.src.crews.rag_crew.rag_crew import RAGCrew
from financial_agent_crewai.src.crews.report_crew.report_crew import ReportCrew
from financial_agent_crewai.src.crews.sec_edgar_crew.sec_edgar_crew import SECEdgarCrew
from financial_agent_crewai.src.crews.yfinance_stock_crew.yfinance_stock_crew import YFinanceStockCrew
from financial_agent_crewai.src.tools.sec_edgar_tools import SecEdgarFilingsInputsList
from financial_agent_crewai.src.utils.constants import CACHE_DIR, COMPARISON_QUERY, USER_QUERY
from financial_agent_crewai.src.utils.utilities import clear_directory

warnings.filterwarnings('ignore', category=SyntaxWarning, module='pysbd')
load_dotenv()

logger = logging.getLogger(__name__)


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

    def __init__(self, query: str) -> None:
        """Initialize the finance flow with research, RAG, and content creation crews."""
        super().__init__()
        self.research_crew = GenericResearchCrew().crew()
        self.query = query
        # self.content_crew = EduContentWriterCrew().crew()
        # Create the cache directory if it does not exist
        os.makedirs(str(CACHE_DIR), exist_ok=True)

        # Empty cache directory
        clear_directory(str(CACHE_DIR))

    @start()  # type: ignore
    def query_decomposition(self) -> Any:
        """
        Decompose the user query.
        """

        query_list = DecompositionCrew().crew().kickoff(inputs={'query': self.query}).pydantic.queries_list

        return query_list

    @listen(query_decomposition)  # type: ignore
    def information_extraction(self, query_list: List[str]) -> Any:
        """
        Decompose the user query.
        """
        # Concatenate the sub-queries into a long query
        decomposed_query = '.'.join(query_list)

        company_input_list = (
            InformationExtractionCrew().crew().kickoff(inputs={'query': decomposed_query}).pydantic.inputs_list
        )

        return company_input_list

    # @start()  # type: ignore
    # def generic_research(self) -> Any:
    #     """
    #     Perform a generic research on the user query.
    #     """

    #     return self.research_crew.kickoff(inputs={'topic': self.query})

    @listen(information_extraction)  # type: ignore
    def sec_edgar(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
        """
        Retrieve the relevant SEC Edgar filings and perform RAG on the user query.
        """
        # Initialize an empty text file
        global_sec_filename = 'global_report_sec.txt'

        sec_reports_list = list()
        for filing_metadata in sec_edgar_inputs_list:
            filename = (
                SECEdgarCrew(input_variables=filing_metadata)  # type: ignore
                .crew()
                .kickoff(
                    {
                        'query': filing_metadata.query,  # type: ignore
                    },
                )
                .pydantic.filename
            )

            sec_reports_list.append(
                RAGCrew(filename=filename)
                .crew()
                .kickoff(
                    {
                        'query': filing_metadata.query,  # type: ignore
                    },
                )
            )

            try:
                # Open the source file in read mode and target file in append mode
                with open(filename, 'r') as source:
                    content = source.read()

                # Concatenate the text from the SEC reports
                with open(global_sec_filename, 'a') as target:
                    target.write(content)

            except FileNotFoundError:
                logger.warning('One of the files was not found. Please check the file paths.')
            except Exception as e:
                logger.warning(f'An error occurred: {e}')

        if len(sec_reports_list) > 0:
            RAGCrew(filename=global_sec_filename).crew().kickoff(
                {
                    'query': COMPARISON_QUERY,
                },
            )
            sec_reports_list.append(global_sec_filename)

        return sec_reports_list

    # @listen(information_extraction)  # type: ignore
    # def yfinance_news(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
    #     """
    #     Retrieve relevant news articles from Yahoo Finance for a particular company
    #     """

    #     symbol_list = [filing_metadata.ticker_symbol for filing_metadata in sec_edgar_inputs_list]  # type: ignore
    #     query_list = [filing_metadata.query for filing_metadata in sec_edgar_inputs_list]  # type: ignore

    #     symbol_list_yfinance_news: List[str] = list()
    #     yfinace_news_reports_list: List[str] = list()
    #     global_yfinance_filename = 'global_report_yfinance_news.txt'
    #     for symbol, query in zip(symbol_list, query_list):
    #         if symbol not in symbol_list_yfinance_news:
    #             symbol_list_yfinance_news.append(symbol)
    #         else:
    #             continue

    #             filenames_list = (
    #                 YahooFinanceNewsCrew(ticker_symbol=symbol)
    #                 .crew()
    #                 .kickoff(
    #                     {
    #                         'query': query,
    #                     },
    #                 )
    #                 .pydantic.filenames
    #             )

    #             for filename in filenames_list:
    #                 yfinace_news_reports_list.append(
    #                     RAGCrew(filename=filename)
    #                     .crew()
    #                     .kickoff(
    #                         {
    #                             'query': query,
    #                         },
    #                     )
    #                 )

    #                 try:
    #                     # Open the source file in read mode and target file in append mode
    #                     with open(filename, 'r') as source:
    #                         content = source.read()

    #                     # Concatenate the text from the SEC reports
    #                     with open(global_yfinance_filename, 'a') as target:
    #                         target.write(content)

    #                 except FileNotFoundError:
    #                     logger.warning('One of the files was not found. Please check the file paths.')
    #                 except Exception as e:
    #                     logger.warning(f'An error occurred: {e}')

    #     if len(yfinace_news_reports_list) > 0:
    #         RAGCrew(filename=global_yfinance_filename).crew().kickoff(
    #             {
    #                 'query': COMPARISON_QUERY,
    #             },
    #         )
    #         yfinace_news_reports_list.append(global_yfinance_filename)

    #     return yfinace_news_reports_list

    @listen(information_extraction)  # type: ignore
    def yfinance_stock_analysis(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
        """
        Retrieve the relevant SEC Edgar filings and perform RAG on the user query.
        """
        # Initialize an empty text file
        global_sec_filename = 'global_report_yfinance.txt'

        yfinance_reports_list: List[str] = list()
        for filing_metadata in sec_edgar_inputs_list:
            filename = (
                YFinanceStockCrew(input_variables=filing_metadata)  # type: ignore
                .crew()
                .kickoff(
                    {
                        'query': filing_metadata.query,  # type: ignore
                    },
                )
                .pydantic.filename
            )

            try:
                # Open the source file in read mode and target file in append mode
                with open(filename, 'r') as source:
                    content = source.read()

                # Concatenate the text from the SEC reports
                with open(global_sec_filename, 'a') as target:
                    target.write(content)

            except FileNotFoundError:
                logger.warning('One of the files was not found. Please check the file paths.')
            except Exception as e:
                logger.warning(f'An error occurred: {e}')

        if len(yfinance_reports_list) > 0:
            RAGCrew(filename=global_sec_filename).crew().kickoff(
                {
                    'query': COMPARISON_QUERY,
                },
            )
            yfinance_reports_list.append(global_sec_filename)

        return yfinance_reports_list

    @listen(and_('generic_research', 'sec_edgar', 'yfinance_news', 'yfinance_stock_analysis'))  # type: ignore
    def report_writing(
        self,
        generic_report: str,
        sec_reports_list: List[str],
        yfinace_news_reports_list: List[str],
        yfinance_reports_list: List[str],
    ) -> Any:
        return (
            ReportCrew(source_path='xxx')
            .crew()
            .kickoff(
                {
                    'query': self.query,
                },
            )
        )


def kickoff() -> None:
    """Initialize and start the educational content generation process."""
    finance_flow = FinancialFlow(
        query=USER_QUERY,
    )
    finance_flow.kickoff()


def plot() -> None:
    """Generate and display a visualization of the flow structure."""
    finance_flow = FinancialFlow(query=USER_QUERY)
    plot_filename = str(CACHE_DIR / 'flow')
    finance_flow.plot(filename=plot_filename)


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
