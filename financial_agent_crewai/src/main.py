"""
Main module for the Educational Content Generation Flow.

This module implements a workflow for generating educational content using
multiple specialized AI crews. It handles the coordination between research
and content creation phases.
"""

import logging
import os
import warnings
from typing import Any, List, Optional

from crewai import LLM
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
from financial_agent_crewai.src.crews.yfinance_news_crew.yfinance_news_crew import YahooFinanceNewsCrew
from financial_agent_crewai.src.crews.yfinance_stocks_crew.yfinance_stocks_crew import YFinanceStockCrew
from financial_agent_crewai.src.tools.general_tools import SubQueriesList
from financial_agent_crewai.src.tools.sec_edgar_tools import SecEdgarFilingsInput, SecEdgarFilingsInputsList
from financial_agent_crewai.src.utils.config import *
from financial_agent_crewai.src.utils.utilities import clear_directory
from utils.model_wrappers.api_gateway import APIGateway

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

    def __init__(
        self,
        query: str,
        source_generic_search: Optional[bool] = None,
        source_sec_filings: Optional[bool] = None,
        source_yfinance_news: Optional[bool] = None,
        source_yfinance_stocks: Optional[bool] = None,
        model_name: str = 'sambanova/Meta-Llama-3.1-70B-Instruct',
        cache_path: Optional[str] = None,
    ) -> None:
        """Initialize the finance flow with research, RAG, and content creation crews."""
        super().__init__()

        self.query = query
        self.source_generic_search = source_generic_search
        self.source_sec_filings = source_sec_filings
        self.source_yfinance_news = source_yfinance_news
        self.source_yfinance_stocks = source_yfinance_stocks
        self.model_name = model_name
        self.report_list: List[str] = list()
        self.generic_report_name = str(CACHE_DIR / 'report_generic_search.txt')

        self.llm = LLM(model=f'sambanova/{model_name}', temperature=TEMPERATURE)

        self.rag_llm = APIGateway.load_llm(
            type='sncloud',
            streaming=False,
            bundle=True,
            do_sample=False,
            max_tokens_to_generate=1024,
            temperature=TEMPERATURE,
            select_expert=RAG_MODEL,
            process_prompt=False,
            sambanova_api_key=os.getenv('SAMBANOVA_API_KEY'),
        )

        self.research_crew = GenericResearchCrew(
            llm=LLM(model=GENERIC_RESEARCH_MODEL, temperature=TEMPERATURE), filename=self.generic_report_name
        ).crew()

        # self.content_crew = EduContentWriterCrew().crew()
        # Create the cache directory if it does not exist
        os.makedirs(str(CACHE_DIR), exist_ok=True)

        # Empty cache directory
        clear_directory(str(CACHE_DIR))

    @start()  # type: ignore
    def generic_research(self) -> Any:
        """
        Perform a generic research on the user query.
        """
        if self.source_generic_search:
            self.report_list.extend(self.generic_report_name)
            return self.research_crew.kickoff(inputs={'query': self.query})
        else:
            return ''

    @start()  # type: ignore
    def query_decomposition(self) -> Any:
        """
        Decompose the user query.
        """
        if self.source_sec_filings or self.source_yfinance_news or self.source_yfinance_stocks:
            query_list = (
                DecompositionCrew(llm=LLM(model=DECOMPOSITION_MODEL, temperature=TEMPERATURE))
                .crew()
                .kickoff(inputs={'query': self.query})
                .pydantic.queries_list
            )
            return query_list
        else:
            return SubQueriesList(queries_list=[self.query])

    @listen(query_decomposition)  # type: ignore
    def information_extraction(self, query_list: List[str]) -> Any:
        """
        Decompose the user query.
        """
        if self.source_sec_filings or self.source_yfinance_news or self.source_yfinance_stocks:
            # Concatenate the sub-queries into a long query
            decomposed_query = '.'.join(query_list)

            company_input_list = (
                InformationExtractionCrew(llm=LLM(model=INFORMATION_EXTRACTION_MODEL, temperature=TEMPERATURE))
                .crew()
                .kickoff(inputs={'query': decomposed_query})
                .pydantic.inputs_list
            )

            return company_input_list
        else:
            SecEdgarFilingsInputsList(
                inputs_list=[
                    SecEdgarFilingsInput(
                        ticker_symbol='',
                        is_relevant_sec=False,
                        filing_type='',
                        filing_quarter=None,
                        year=2024,
                        query=self.query,
                    )
                ]
            )

    @listen(information_extraction)  # type: ignore
    def sec_edgar(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
        """
        Retrieve the relevant SEC Edgar filings and perform RAG on the user query.
        """
        if self.source_sec_filings:
            # Initialize an empty text file
            global_sec_filename = 'global_report_sec.txt'

            sec_reports_list = list()
            for filing_metadata in sec_edgar_inputs_list:
                filename = (
                    SECEdgarCrew(
                        llm=LLM(model=SEC_EDGAR_MODEL, temperature=TEMPERATURE),
                        input_variables=filing_metadata,  # type: ignore
                    )
                    .crew()
                    .kickoff(
                        {
                            'query': filing_metadata.query,  # type: ignore
                        },
                    )
                    .pydantic.filename
                )

                sec_reports_list.append(filename)

                RAGCrew(
                    filename=filename,
                    llm=LLM(model=RAG_MODEL, temperature=TEMPERATURE),
                    rag_llm=self.rag_llm,
                ).crew().kickoff(
                    {'query': filing_metadata.query},  # type: ignore
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

            if len(sec_reports_list) > 1:
                RAGCrew(
                    filename=global_sec_filename,
                    llm=LLM(model=RAG_MODEL, temperature=TEMPERATURE),
                    rag_llm=self.rag_llm,
                ).crew().kickoff(
                    {'query': COMPARISON_QUERY},
                )
                sec_reports_list.append(global_sec_filename)

            # Append the list of SEC Edgar reports
            self.report_list.extend(sec_reports_list)
            return sec_reports_list
        else:
            return list()

    @listen(information_extraction)  # type: ignore
    def yfinance_news(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
        """
        Retrieve relevant news articles from Yahoo Finance for a particular company
        """
        if self.source_yfinance_news:
            symbol_list = [filing_metadata.ticker_symbol for filing_metadata in sec_edgar_inputs_list]  # type: ignore
            query_list = [filing_metadata.query for filing_metadata in sec_edgar_inputs_list]  # type: ignore

            yfinace_news_reports_list: List[str] = list()
            global_yfinance_filename = 'global_report_yfinance_news.txt'
            for symbol, query in zip(symbol_list, query_list):
                current_filenames_list = (
                    YahooFinanceNewsCrew(llm=LLM(model=YFINANCE_NEWS_MODEL, temperature=TEMPERATURE))
                    .crew()
                    .kickoff(
                        {'ticker_symbol': symbol},
                    )
                    .pydantic.filenames
                )

                for filename in current_filenames_list:
                    RAGCrew(
                        filename=filename,
                        llm=LLM(model=RAG_MODEL, temperature=TEMPERATURE),
                        rag_llm=self.rag_llm,
                    ).crew().kickoff(
                        {'query': query},
                    )

                    try:
                        # Open the source file in read mode and target file in append mode
                        with open(filename, 'r') as source:
                            content = source.read()

                        # Concatenate the text from the SEC reports
                        with open(global_yfinance_filename, 'a') as target:
                            target.write(content)

                        yfinace_news_reports_list.append(filename)

                    except FileNotFoundError:
                        logger.warning('One of the files was not found. Please check the file paths.')
                    except Exception as e:
                        logger.warning(f'An error occurred: {e}')

            if len(yfinace_news_reports_list) > 1:
                RAGCrew(
                    filename=global_yfinance_filename,
                    llm=LLM(model=RAG_MODEL, temperature=TEMPERATURE),
                    rag_llm=self.rag_llm,
                ).crew().kickoff(
                    {'query': COMPARISON_QUERY},
                )
                yfinace_news_reports_list.append(global_yfinance_filename)

                # Append the list of SEC Edgar reports
                self.report_list.extend(yfinace_news_reports_list)

                return yfinace_news_reports_list
        else:
            return list()

    @listen(information_extraction)  # type: ignore
    def yfinance_stocks(self, sec_edgar_inputs_list: SecEdgarFilingsInputsList) -> List[str]:
        """
        Retrieve the relevant SEC Edgar filings and perform RAG on the user query.
        """
        if self.source_yfinance_stocks:
            # Initialize an empty text file
            global_sec_filename = 'global_report_yfinance.txt'

            yfinance_reports_list: List[str] = list()
            for filing_metadata in sec_edgar_inputs_list:
                filename = (
                    YFinanceStockCrew(
                        input_variables=filing_metadata,  # type: ignore
                        llm=LLM(model=YFINANCE_STOCKS_MODEL, temperature=TEMPERATURE),
                    )
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

            if len(yfinance_reports_list) > 1:
                RAGCrew(
                    filename=global_sec_filename,
                    llm=LLM(model=RAG_MODEL, temperature=TEMPERATURE),
                    rag_llm=self.rag_llm,
                ).crew().kickoff(
                    {'query': COMPARISON_QUERY},
                )
                yfinance_reports_list.append(global_sec_filename)

            # Append the list of SEC Edgar reports
            self.report_list.extend(yfinance_reports_list)
            return yfinance_reports_list
        else:
            return list()

    @listen(and_(generic_research, sec_edgar, yfinance_news, yfinance_stocks))  # type: ignore
    def report_writing(
        self,
    ) -> Any:
        """Write the final financial report."""
        for report in self.report_list:
            # Load the text file
            with open(report, 'r') as f:
                report_txt = f.read()
            ReportCrew(
                llm=LLM(model=REPORT_MODEL, temperature=TEMPERATURE),
            ).crew().kickoff(
                {
                    'section': self.query,
                },
            )


def run() -> None:
    """Initialize and start the educational content generation process."""
    finance_flow = FinancialFlow(
        query=USER_QUERY,
        source_generic_search=SOURCE_GENERIC_SEARCH,
        source_sec_filings=SOURCE_SEC_FILINGS,
        source_yfinance_news=SOURCE_YFINANCE_NEWS,
        source_yfinance_stocks=SOURCE_YFINANCE_STOCK,
        model_name='Meta-Llama-3.1-70B-Instruct',
    )
    finance_flow.kickoff()


def plot() -> None:
    """Generate and display a visualization of the flow structure."""
    finance_flow = FinancialFlow(query=USER_QUERY)
    plot_filename = str(CACHE_DIR / 'flow')
    finance_flow.plot(filename=plot_filename)


if __name__ == '__main__':
    plot()
    run()
