import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit
from fpdf import FPDF
from fpdf.fpdf import Align
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_assistant.src.retrieval import get_qa_response
from financial_assistant.src.tools import coerce_str_to_list, time_llm
from financial_assistant.streamlit.constants import *

EMPTY_TEXT_PLACEHOLDER = 'Empty text content'

BACKGROUND_COLOUR = (255, 229, 180)
L_MARGIN = 15
T_MARGIN = 20
LOGO_WIDTH = 25


FONT = 'helvetica'


class PDFReport(FPDF):  # type: ignore
    """Class for generating PDF reports."""

    def header(self, title_name: str = 'Financial Report') -> None:
        """Overrides the FPDF header method."""
        self.set_text_color(SAMBANOVA_ORANGE)

        # Rendering logo:
        self.image(
            'https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
            self.w - self.l_margin - LOGO_WIDTH,
            self.t_margin - self.t_margin / 2,
            LOGO_WIDTH,
        )

        # Setting font
        self.set_font(FONT, 'B', 16)
        # Printing title
        self.cell(0, 10, title_name, align=Align.C)
        self.ln(20)
        self.set_font(FONT, '', 10)
        self.cell(
            0,
            0,
            'Powered by SambaNova',
            align=Align.R,
        )
        # Performing a line break:
        self.ln(5)

    def footer(self) -> None:
        """Overrides the FPDF footer method."""
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        self.set_text_color(128)
        # Setting font
        self.set_font(FONT, 'I', 8)
        # Printing page number:
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title: str) -> None:
        """Writes the chapter title."""
        self.set_text_color(SAMBANOVA_ORANGE)
        self.set_font(FONT, 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_summary(
        self,
        body: str,
    ) -> None:
        """Writes the chapter summary."""
        self.set_text_color((0, 0, 0))
        self.set_font(FONT, 'I', 11)
        self.multi_cell(0, 5, body)
        self.ln(5)

    def chapter_body(
        self,
        body: str,
    ) -> None:
        """Writes the chapter body."""
        self.set_text_color((0, 0, 0))
        self.set_font(FONT, '', 11)
        self.multi_cell(0, 5, body)
        self.ln(5)

    def add_figure(self, figure_path: str) -> None:
        """Adds a figure to the PDF."""
        # Calculate the desired width of the figure (90% of the page width)
        page_width = self.w - 2 * self.l_margin
        figure_width = page_width * 0.9

        # Place the image on the PDF with the calculated width
        self.image(figure_path, x=self.l_margin + (page_width - figure_width) / 2, w=figure_width)
        self.ln(10)


def read_txt_files(directory: str) -> List[str]:
    """Reads all the text files from a directory."""
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
    return texts


def parse_documents(documents: List[str]) -> List[Tuple[str, Any]]:
    """
    Parse the documents into a format that is easier to work with.

    Args:
        documents: A list of text files.
    Returns:
        A list of tuples, each containing the following elements:
            - The text.
            - The paths of any `png` figures.
    """
    report_content: List[Tuple[str, List[str] | None]] = list()
    figure_regex = re.compile(r'\/\b\S+\.png\b')
    endline_regex = re.compile(r'\n\n')
    startline_regex = re.compile(r'^\n\n')
    selectd_tables_regex = re.compile(r'Table')
    colon_space_regex = re.compile(r': , ')
    final_colon_space_regex = re.compile(r': \.')

    for doc in documents:
        parts = doc.split('\n\n\n\n')
        for part in parts:
            # Search for figure paths
            figure_matches = figure_regex.findall(part)
            if len(figure_matches) > 0:
                # Remove figure paths from text
                cleaned_part = figure_regex.sub('', part)
                # Clean extra newline
                cleaned_part = endline_regex.sub('', cleaned_part)
                cleaned_part = re.sub(selectd_tables_regex, '\nSelected SQL tables: Table', cleaned_part, count=1)
                cleaned_part = colon_space_regex.sub(', ', cleaned_part)
                cleaned_part = final_colon_space_regex.sub('.', cleaned_part)
            else:
                # Clean extra newline
                cleaned_part = startline_regex.sub('', part)

            if figure_matches:
                report_content.append((cleaned_part.strip(), figure_matches))
            else:
                report_content.append((cleaned_part.strip(), None))

    return report_content


# Define your desired data structure.
class SectionTitleSummary(BaseModel):
    """Model representing the title and summary of a section."""

    title: str = Field(..., description='Title of the section.')
    summary: str = Field(..., description='Summary of the section.')


class SectionTitle(BaseModel):
    """Model representing the title of a section."""

    title: str = Field(..., description='The title of the section.')


def generate_pdf(
    report_content: List[Tuple[str, Optional[str]]],
    output_file: str,
    title_name: str,
    include_summary: bool = False,
) -> bytes:
    """
    Generate a PDF report from the given parsed content.

    Args:
        report_content: The parsed content as a list of tuples with text and figure paths.
        output_file: The path to the output file.
        title_name: The name of the PDF report.
        include_summary: Whether or not to include a summary at the end of each section
            and general abstract and summary at the beginning of the report.
    """
    pdf = PDFReport()
    pdf.set_font(FONT)
    pdf.set_page_background(BACKGROUND_COLOUR)
    pdf.set_margins(L_MARGIN, T_MARGIN)

    title_generation_template = (
        'Generate a json formatted concise title (less than 10 words) '
        + 'that captures and summarizes the main idea or theme of following paragraphs.'
        + '\nParagraphs:{text}.'
        + '\n{format_instructions}'
    )

    pdf.add_page()
    content_list: List[Dict[str, str | List[str]]] = list()
    for idx, content in enumerate(report_content):
        text, figure_paths = content
        if (text is None and figure_paths is None) or (text is not None and len(text) == 0 and figure_paths is None):
            continue

        # Clean the text to conform to unicode standards
        content_list.append(
            {
                'text': clean_unicode_text(text),
                'figure_path': figure_paths if figure_paths is not None else list(),
            }
        )

    # Parse and split the documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=0)
    docs = list()

    # Compose the documents
    for content_item in content_list:
        text_item = content_item['text']
        if isinstance(text_item, str) and len(text_item) > 0:
            docs.append(Document(page_content=text_item))
        else:
            docs.append(Document(page_content=EMPTY_TEXT_PLACEHOLDER))

    # Split the documents
    split_docs = text_splitter.split_documents(docs)

    # Add a summary at the beginning
    if include_summary:
        progress_text = f'Summarizing {len(content_list)} queries...'
        summary_bar = streamlit.progress(0, text=progress_text)

        intermediate_summaries, intermediate_titles, final_summary, abstract = summarize_text(split_docs)
        if len(abstract) > 0:
            pdf.chapter_title('Abstract')
            pdf.chapter_summary(abstract)

        if len(final_summary) > 0:
            pdf.chapter_title('Summary')
            pdf.chapter_summary(final_summary)

        for idx, item in enumerate(content_list):
            time.sleep(0.01)
            summary_bar.progress(idx + 1, text=progress_text)

            # Add the section title
            if len(intermediate_titles[idx]) > 0:
                pdf.chapter_title(intermediate_titles[idx])
            else:
                pdf.chapter_title('Section ' + str(idx + 1))

            # Add the section summary
            if len(intermediate_summaries[idx]) > 0:
                pdf.chapter_summary(intermediate_summaries[idx])

            # Add the section body
            if item['text'] is not None and item['text'] != EMPTY_TEXT_PLACEHOLDER and isinstance(item['text'], str):
                pdf.chapter_body(item['text'])

            # Add the section figures
            if item['figure_path'] is not None and len(item['figure_path']) > 0:
                for figure in item['figure_path']:
                    pdf.add_figure(figure)

        time.sleep(0.01)
        summary_bar.empty()
    else:
        for idx, item in enumerate(content_list):
            pdf.chapter_title('Query ' + str(idx))
            if item['text'] is not None and item['text'] != EMPTY_TEXT_PLACEHOLDER and isinstance(item['text'], str):
                pdf.chapter_body(item['text'])
            if item['figure_path'] is not None and len((item['figure_path'])) > 0:
                for figure in item['figure_path']:
                    pdf.add_figure(figure)

    pdf.output(output_file)
    return bytes(pdf.output())


class Summary(BaseModel):
    """Model representing the title and summary of a document."""

    title: str = Field(..., description='The title of the document.')
    summary: str = Field(..., description='The extracted summary of the document.')


class ReduceSummary(BaseModel):
    """Model representing the final concise summary of the documents."""

    summary: str = Field(..., description='The final concise summary of the documents.')


def summarize_text(split_docs: List[Document]) -> Tuple[List[str], List[str], str, str]:
    """
    Summarize the text in `split_docs` using the LLM.

    Args:
        split_docs: List of documents to summarize.
    Returns:
        A tuple containing the following elements:
            - Intermediate summaries of each section.
            - Intermediate= titles of each section.
            - Final summary of the document.
            - Abstract of the document.
    """
    # Extract intermediate titles and summaries for each document in the split docs
    intermediate_results = list()
    for doc in split_docs:
        try:
            intermediate_results.append(invoke_summary_map_chain(doc))
        except:
            intermediate_results.append(Summary(title='', summary=''))

    intermediate_summaries = [item.summary for item in intermediate_results]
    intermediate_titles = [item.title for item in intermediate_results]

    # Extract final summary from intermediate summaries
    try:
        final_summary = invoke_reduction_chain(intermediate_summaries)
    except:
        final_summary = ''

    # Extract abstract from the final summary
    try:
        abstract = invoke_abstract_chain(final_summary)
    except:
        abstract = ''

    return intermediate_summaries, intermediate_titles, final_summary, abstract


@time_llm
def invoke_abstract_chain(final_summary: str) -> Any:
    """Invoke the LLM to extract an abstract for the final summary."""

    # Extract the LLM
    llm = streamlit.session_state.llm.llm

    # Abstract parser
    abstract_parser = PydanticOutputParser(pydantic_object=ReduceSummary)  # type: ignore

    # Abstract template
    abstract_template = """Write a concise summary of the following:
        {final_summary}.\n
        {format_instructions}
        """

    # Abstract prompt
    abstact_prompt = PromptTemplate(
        template=abstract_template,
        input_variables=['final_summary'],
        partial_variables={'format_instructions': abstract_parser.get_format_instructions()},
    )

    # Abstract chain
    abstract_chain = abstact_prompt | llm | abstract_parser

    # Run chain
    abstract = abstract_chain.invoke(final_summary).summary

    return abstract


@time_llm
def invoke_reduction_chain(intermediate_summaries: List[str]) -> Any:
    """Invoke the LLM to reduce a list of intermediate summaries to one final summary."""
    # Extract the LLM
    llm = streamlit.session_state.llm.llm

    # Reduce parser
    reduce_parser = PydanticOutputParser(pydantic_object=ReduceSummary)  # type: ignore

    # Reduce template
    reduce_template = """The following is set of summaries:
        {intermediate_summaries}
        Take these and distill it into a final, consolidated summary of the main themes.\n'
        '{format_instructions}'
        """

    # Reduce prompt
    reduce_prompt = PromptTemplate(
        template=reduce_template,
        input_variables=['intermediate_summaries'],
        partial_variables={'format_instructions': reduce_parser.get_format_instructions()},
    )

    # Reduce chain
    reduce_chain = reduce_prompt | llm | reduce_parser

    # Run chain
    final_summary = reduce_chain.invoke('\n'.join(intermediate_summaries)).summary

    return final_summary


@time_llm
def invoke_summary_map_chain(doc: Document) -> Any:
    """Invoke the LLM to summarize the text in `doc` using the LLM."""

    # Extract the LLM
    llm = streamlit.session_state.llm.llm

    # Map parser
    map_parser = PydanticOutputParser(pydantic_object=Summary)  # type: ignore

    # Map template
    map_template = """The following is a document:
        {doc}
        Please identify the main theme (title + summary) of this document.\n.
        {format_instructions}
        """

    # Map prompt
    map_prompt = PromptTemplate(
        template=map_template,
        input_variables=['doc'],
        partial_variables={'format_instructions': map_parser.get_format_instructions()},
    )

    # Map chain
    map_chain = map_prompt | llm | map_parser

    return map_chain.invoke(doc)


class PDFRAGInput(BaseModel):
    """Tool retrieving the provided PDF file(s) to answer the user query via Retrieval-Augmented Generation (RAG)."""

    user_query: str = Field(..., description='The user query.')
    pdf_files_names: List[str] | str = Field(
        ..., description='The list of paths to the PDF file(s) to be used for RAG.'
    )


@tool(args_schema=PDFRAGInput)
def pdf_rag(user_query: str, pdf_files_names: List[str] | str) -> Any:
    """
    Tool retrieving the provided PDF file(s) to answer the user query via Retrieval-Augmented Generation (RAG).

    Args:
        user_query: The user query.
        pdf_files: The path to the PDF file to be used for RAG.

    Returns:
        The answer to the user query, generated using RAG.
    """
    # Check inputs
    assert isinstance(user_query, str), 'The user query must be a string.'
    assert isinstance(pdf_files_names, (list, str)), 'The PDF files must be a list of paths or a path string.'

    # If `symbol_list` is a string, coerce it to a list of strings
    pdf_files_names = coerce_str_to_list(pdf_files_names)

    assert all(isinstance(file, str) for file in pdf_files_names), 'The PDF files must be a list of path strings.'

    # Load PDF files
    documents = []
    for file in pdf_files_names:
        pdf_path = os.path.join(streamlit.session_state.pdf_generation_directory, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    # Instantiate the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # Split documents into chunks
    chunked_documents = text_splitter.split_documents(documents)

    # Retrieve the QA response
    response = get_qa_response(user_query, chunked_documents)['answer']

    return response


def clean_unicode_text(text: str) -> str:
    """Clean the text by excluding non unicode characters."""

    # This pattern matches any character that is a Unicode letter, digit, punctuation, or space
    pattern = re.compile(r'[^\w\s.,!?\'¿¡"@#$%^&*()_+={}|[\]\\;\-:"<>?/`~]')

    return re.sub(pattern, '', text)
