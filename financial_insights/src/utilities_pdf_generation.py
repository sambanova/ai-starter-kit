import os
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit
from fpdf import FPDF
from fpdf.fpdf import Align
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from financial_insights.streamlit.constants import *

BACKGROUND_COLOUR = (255, 229, 180)
L_MARGIN = 15
T_MARGIN = 20
LOGO_WIDTH = 25


FONT_SIZE = 'helvetica'


class PDFReport(FPDF):  # type: ignore
    def header(self, title_name: str = 'Financial Report') -> None:
        self.set_text_color(SAMBANOVA_ORANGE)

        # Rendering logo:
        self.image(
            'https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
            self.w - self.l_margin - LOGO_WIDTH,
            self.t_margin - self.t_margin / 2,
            LOGO_WIDTH,
        )

        # Setting font: helvetica bold 15
        self.set_font(FONT_SIZE, 'B', 15)
        # Printing title:
        self.cell(0, 10, title_name, align=Align.C)
        self.ln(LOGO_WIDTH + 1)
        self.set_font(FONT_SIZE, '', 10)
        self.cell(
            0,
            0,
            'Powered by SambaNova Finance App',
            align=Align.R,
        )
        # Performing a line break:
        self.ln(5)

    def footer(self) -> None:
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        self.set_text_color(128)
        # Setting font: helvetica italic 8
        self.set_font(FONT_SIZE, 'I', 8)
        # Printing page number:
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title: str) -> None:
        self.set_text_color(SAMBANOVA_ORANGE)
        self.set_font(FONT_SIZE, 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_summary(
        self,
        body: str,
    ) -> None:
        self.set_text_color((0, 0, 0))
        self.set_font(FONT_SIZE, 'I', 11)
        self.multi_cell(0, 5, body)
        self.ln(5)

    def chapter_body(
        self,
        body: str,
    ) -> None:
        self.set_text_color((0, 0, 0))
        self.set_font(FONT_SIZE, '', 11)
        self.multi_cell(0, 5, body)
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
        parts = doc.split('\n\n\n\n')
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
class SectionTitleSummary(BaseModel):
    title: str = Field(description='Title of the section.')
    summary: str = Field(description='Summary of the section.')


class SectionTitle(BaseModel):
    title: str = Field(description='Title of the section.')


def generate_pdf(
    report_content: List[Tuple[str, Optional[str]]],
    output_file: str,
    title_name: str,
    include_summary: bool = False,
) -> None:
    pdf = PDFReport()
    pdf.set_font('Helvetica')
    pdf.set_page_background(BACKGROUND_COLOUR)
    pdf.set_margins(L_MARGIN, T_MARGIN)

    if include_summary:
        # Set up a parser + inject instructions into the prompt template.
        title_parser = PydanticOutputParser(pydantic_object=SectionTitleSummary)  # type: ignore

        title_generation_template = (
            'Generate a json formatted concise title (less than 10 words) and a brief summary (less than 2 sentences) '
            + 'that respectively capture and summarize the main idea or theme of following paragraphs.'
            + '\nParagraphs:{text}.'
            + '\n{format_instructions}'
        )
    else:
        # Set up a parser + inject instructions into the prompt template.
        title_parser = PydanticOutputParser(pydantic_object=SectionTitle)  # type: ignore

        title_generation_template = (
            'Generate a json formatted concise title (less than 10 words) '
            + 'that captures and summarizes the main idea or theme of following paragraphs.'
            + '\nParagraphs:{text}.'
            + '\n{format_instructions}'
        )

    title_prompt = PromptTemplate(
        template=title_generation_template,
        input_variables=['text'],
        partial_variables={'format_instructions': title_parser.get_format_instructions()},
    )

    title_chain = title_prompt | streamlit.session_state.fc.llm | title_parser

    pdf.add_page()
    summaries_list = list()
    content_list: List[Dict[str, str]] = list()
    for idx, content in enumerate(report_content):
        content_list.append(dict())
        text, figure_path = content

        if text:
            try:
                # Extract title from text using the state LLM
                answer = title_chain.invoke({'text': text})
                extracted_title = answer.title
                
                if include_summary:
                    extracted_summary = answer.summary
                    content_list[idx]['summary'] = extracted_summary
                
                
            except Exception as e:
                extracted_title = 'Query'
                extracted_summary = ''
        
            content_list[idx]['title'] = extracted_title
            summaries_list.append(extracted_summary)
            content_list[idx]['text'] = text
            
        if figure_path:
            content_list[idx]['figure_path'] = figure_path
            pdf.add_figure(figure_path)

    if include_summary:
        pdf.chapter_title('Summary')
        for summary in summaries_list:
            pdf.chapter_summary(summary)

    for item in content_list:
        pdf.chapter_title(item['title'])
        if include_summary:
            pdf.chapter_summary(item['summary'])
        pdf.chapter_body(item['text'])

    pdf.output(output_file)
