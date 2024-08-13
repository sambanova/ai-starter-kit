import logging
import os
import re
import sys
from typing import Any, List, Optional, Tuple

import streamlit
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


TEMP_DIR = 'financial_insights/streamlit/cache/'
SOURCE_DIR = 'financial_insights/streamlit/cache/sources/'
CONFIG_PATH = 'financial_insights/config.yaml'


import os

from fpdf import FPDF


class PDFReport(FPDF):  # type: ignore
    def header(self, title_name: str = 'Financial Report') -> None:
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title_name, 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title: str) -> None:
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body: str) -> None:
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln(5)

    def add_figure(self, figure_path: str) -> None:
        # Calculate the desired width of the figure (90% of the page width)
        page_width = self.w - 2 * self.l_margin
        figure_width = page_width * 0.9

        # Place the image on the PDF with the calculated width
        self.image(figure_path, x=self.l_margin + (page_width - figure_width) / 2, w=figure_width)
        self.ln(10)


def read_txt_files(directory: str) -> List[str]:
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
    return texts


def parse_documents(documents: List[str]) -> list[tuple[str, Any]]:
    report_content = []
    figure_regex = re.compile(r'financial_insights/*[^\s]+\.png')

    for doc in documents:
        parts = doc.split('\n\n')
        for part in parts:
            # Search for figure paths
            figure_matches = figure_regex.findall(part)
            cleaned_part = figure_regex.sub('', part)  # remove figure paths from text

            if figure_matches:
                for figure_path in figure_matches:
                    report_content.append((cleaned_part.strip(), figure_path))
            else:
                report_content.append((cleaned_part.strip(), None))

    return report_content


# Define your desired data structure.
class SectionTitle(BaseModel):
    title: str = Field(description='Title of the section.')


def generate_pdf(report_content: List[Tuple[str, Optional[str]]], output_file: str, title_name: str) -> None:
    pdf = PDFReport()

    # Set up a parser + inject instructions into the prompt template.
    title_parser = PydanticOutputParser(pydantic_object=SectionTitle)  # type: ignore

    title_generation_template = (
        'Generate a concise title (less than 10 words) '
        + 'that summarizes the main idea or theme of following paragraph.\n'
        + 'Paragraph:{text}.\n{format_instructions}'
    )

    title_prompt = PromptTemplate(
        template=title_generation_template,
        input_variables=['text'],
        partial_variables={'format_instructions': title_parser.get_format_instructions()},
    )

    title_chain = title_prompt | streamlit.session_state.fc.llm | title_parser

    pdf.add_page()
    for content in report_content:
        text, figure_path = content

        if text:
            # Extract title from text using the state LLM
            extracted_title = title_chain.invoke({'text': text}).title
            pdf.chapter_title(extracted_title)
            pdf.chapter_body(text)

        if figure_path:
            pdf.add_figure(figure_path)

    pdf.output(output_file)
