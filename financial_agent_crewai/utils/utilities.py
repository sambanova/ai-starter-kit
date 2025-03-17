import base64
import json
import logging
import os
import re
import shutil
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Generator, List, Match, Optional

import markdown
import pandas
import schedule
import weasyprint  # type: ignore
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def clear_directory(directory: str | Path) -> Optional[str]:
    """
    Clears all contents of the specified directory,
    including all files and subdirectories.

    Args:
        directory: The path to the directory to clear.

    Returns:
        An error message if the directory does not exist or is not a directory,
        otherwise None if the operation is successful.

    Raises:
        OSError: If an error occurs while deleting files or directories.
    """
    if not os.path.exists(directory):
        return f'Directory {directory} does not exist.'

    if not os.path.isdir(directory):
        return f'The path {directory} is not a directory.'

    try:
        # Iterate over the contents of the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            # Remove directory or file
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove subdirectory
            else:
                os.remove(item_path)  # Remove file
    except OSError as e:
        raise OSError(f'Error while clearing directory: {e}')

    return None


def parse_table_str(table_str: str) -> Any:
    """
    Parse a JSON-encoded string into a dictionary containing
    columns, index, and data.

    Args:
        table_str: The raw string that encodes a JSON object with keys:
            - 'columns': list of column names
            - 'index': list of row indices (e.g., timestamps)
            - 'data': nested list representing row data

    Returns:
        A dictionary with keys 'columns', 'index', and 'data'.
    """
    return json.loads(table_str)


def timestamp_to_date_string(ts_millis: float) -> str:
    """
    Convert a millisecond-based Unix timestamp into a human-readable YYYY-MM-DD date string in UTC.
    """
    dt = pandas.to_datetime(ts_millis, unit='ms', utc=True)
    return str(dt.strftime('%Y-%m-%d'))


def dict_to_markdown_table(table_data: Dict[str, Any], title: str) -> str:
    """
    Convert a dictionary containing 'columns', 'index', and 'data'
    to a Markdown-formatted table string.

    Args:
        table_data: A dictionary with keys:
            - 'columns': list of column headers
            - 'index': list of row labels (e.g., timestamps)
            - 'data': list of row values (each a list of floats or strings)
        title: A title or heading for the table in Markdown.

    Returns:
        A string containing the table in Markdown format.
    """
    # Extract the table data
    columns = table_data.get('columns', list())
    index = table_data.get('index', list())
    data = table_data.get('data', list())

    # Start building Markdown
    lines: List[str] = list()

    # Title as a second-level heading
    lines.append(f'## {title}')

    # Header row
    if pandas.to_datetime(pandas.Series(index[0]), unit='ms')[0].to_pydatetime().year >= 2000:
        header = ['Date'] + columns
    else:
        header = ['Index'] + columns
    lines.append('| ' + ' | '.join(header) + ' |')
    # Markdown requires a separator row of dashes
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    # Populate each row
    for row_idx, row_data in enumerate(data):
        if pandas.to_datetime(pandas.Series(index[row_idx]), unit='ms')[0].to_pydatetime().year >= 2000:
            # Convert the timestamp in 'index' to a date string
            index_str = timestamp_to_date_string(index[row_idx])
        else:
            index_str = index[row_idx] if len(data) > 1 else ''

        # Convert each value in row_data to a string
        row_cells = [str(val) for val in row_data]
        lines.append('| ' + index_str + ' | ' + ' | '.join(row_cells) + ' |')

    # Combine all lines into a single string
    return '\n'.join(lines)


def convert_file_of_image_paths_to_markdown(
    input_file: str | List[str], output_file: str | Path, alt_text: str = 'Image'
) -> None:
    """
    Reads an input text file, searches for any .png paths embedded in the text,
    and writes the resulting text to an output file. In the resulting text,
    each found .png path is replaced with a Markdown image reference of the form:
        ![<alt text>](<image path>)

    Args:
        input_file: Path to the input text file.
        output_file: Path to the output file where Markdown statements will be written.
        alt_text: The alt text to use for each image (defaults to "Image").

    Raises:
        TypeError if `input_file` is not a string or a list of strings.

    Example:
        Suppose 'input.txt' has the following content:
            Here is a reference to financial_agent_crewai/cache/yfinance_stocks/262ae6f5.png
            Additional text is here! Also, path/to/another/image.png is included.

        After running:
            convert_file_of_image_paths_to_markdown('input.txt', 'output.md', 'My Image')

        'output.md' will contain:
            Here is a reference to ![My Image](financial_agent_crewai/cache/yfinance_stocks/262ae6f5.png)
            Additional text is here! Also, ![My Image](path/to/another/image.png) is included.
    """
    # For images embedded in a text file
    if isinstance(input_file, str):
        # Read entire file contents
        with open(input_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        # Regex pattern for any sequence of non-whitespace characters ending in .png
        pattern = r'(\S+\.png)'

        # Replace each .png path with the Markdown image reference
        transformed_text = re.sub(pattern, lambda match: f'![{alt_text}]({match.group(1)})', text)
    # For list of strings
    elif isinstance(input_file, list) and all(isinstance(image_path, str) for image_path in input_file):
        transformed_text = ''
        for image_path in input_file:
            image_name = Path(image_path).name.strip('.png')
            transformed_text += f'![{alt_text} {image_name}]({image_path})\n\n'

    else:
        raise TypeError(f'Only strings or lists of strings are supported. Got type {type(input_file)}.')

    # Write the transformed text to the output file
    with open(output_file, 'a', encoding='utf-8') as outfile:
        outfile.write(transformed_text + '\n')


def clean_markdown_content(content: str) -> str:
    """
    Clean Markdown content.

    Convert local Markdown references of the form ![alt text](local_path.png)
    into base64-embedded images so they can display inline.
    Then convert everything to HTML and make minor tidy-ups.
    """

    # Regex to find Markdown image references: ![alt text](image_path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def to_base64(match: Match[str]) -> str:
        alt_text = match.group(1)
        img_path = match.group(2)

        # If it's already a URL (http/https/data), leave it as is.
        if any(img_path.lower().startswith(prefix) for prefix in ('http://', 'https://', 'data:')):
            return match.group(0)

        # If the file doesn't exist on disk, leave the reference as is (broken).
        if not os.path.isfile(img_path):
            return match.group(0)

        # Read and embed the image as base64
        with open(img_path, 'rb') as f:
            img_data = f.read()
        encoded = base64.b64encode(img_data).decode('utf-8')
        # Rebuild the Markdown image tag but with data URI
        return f'![{alt_text}](data:image/png;base64,{encoded})'

    # 1) Convert local images into base64 within the Markdown text
    content = re.sub(pattern, to_base64, content)

    # 2) Convert the entire Markdown to HTML
    html = markdown.markdown(content, extensions=['tables', 'fenced_code', 'toc', 'md_in_html'])

    # 3) Use BeautifulSoup to add classes or further tweak the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Render the tables
    for table in soup.find_all('table'):
        # Get whatever classes the table already has, or an empty list if none
        existing_classes = table.get('class') or list()

        # Use a set so we don't add duplicates
        updated_classes = set(existing_classes).union({'table', 'table-striped', 'table-bordered'})

        # Assign the updated classes back to the table
        table['class'] = list(updated_classes)

    return str(soup)


def convert_html_to_pdf(html_str: str, output_file: Optional[str | Path] = None) -> Any:
    """
    Convert HTML to PDF.

    Convert HTML to PDF while applying page and image scaling rules
    so that large images do not get cut off.
    """
    # Define CSS that scales images and applies basic page sizing
    style = """
    @page {
        size: A4;             /* Adjust if you'd like a different format */
        margin: 2cm;          /* Adjust margins as desired */
    }


    img {
        max-width: 100%;      /* Scale down images that exceed page width */
        height: auto;         /* Preserve aspect ratio */
        display: block;       /* Avoid text wrapping around large images */
    }

    /* Optional supplemental table styling in addition to Bootstrap classes */
    table {
        width: 100%;
        margin-bottom: 1em;   /* Nice spacing after tables */
        border-collapse: collapse;
    }

    table, th, td {
        border: 1px solid #dee2e6;  /* Lightweight border for tables in PDF */
        vertical-align: top;
        padding: 0.75rem;
    }

    /* Example: if you want consistent header background in PDF */
    thead th {
        background-color: #f8f9fa;
        border-bottom: 2px solid #dee2e6;
    }
    """
    # Create a WeasyPrint CSS object from the style string
    stylesheet = weasyprint.CSS(string=style)

    # Build the PDF in-memory, applying the custom stylesheet
    pdf_data = weasyprint.HTML(string=html_str).write_pdf(target=output_file, stylesheets=[stylesheet])

    return pdf_data


def extract_entities(input: str) -> List[Dict[str, Any]]:
    # Remove escape sequences (ANSI codes for formatting) first
    cleaned_input = re.sub(r'\[[0-9;]*m', '', input)

    # Now use regex to extract the agent name and the final answers
    pattern = re.compile(r'# (.*?)\n.*?## (Final Answer):\s*(\{.*?\})', re.DOTALL)

    matches = pattern.findall(cleaned_input)

    result = list()

    for match in matches:
        agent_name = match[0].split(':')[-1].strip()  # Extract agent name

        # Extract full JSON output (with nested JSON objects)
        i = cleaned_input.index(match[2])
        stack = cleaned_input[i]
        json_result = cleaned_input[i]
        i += 1
        while (len(stack) > 0) and (i < len(cleaned_input)):
            json_result += cleaned_input[i]
            stack += '{' if cleaned_input[i] == '{' else ''
            if cleaned_input[i] == '}':
                stack = stack[:-1]
            i += 1

        # Format the result string for each agent
        result.append({'agent_name': agent_name, 'agent_output': json.loads(json_result)})

    return result


@contextmanager
def st_capture(output_func: Any) -> Generator[StringIO, None, None]:
    """
    Context manager for capturing stdout and redirecting to Streamlit.

    Args:
        output_func (Callable[str]): Function to handle captured output.

    Yields:
        StringIO: String buffer containing captured output.
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            # Each time something is written to stdout,
            # we send it to Streamlit via `output_func`.
            agent_outputs = extract_entities(stdout.getvalue())
            output_func(json.dumps(agent_outputs), expanded=2)

            return ret

        stdout.write = new_write  # type: ignore
        yield stdout


def list_first_order_subfolders(directory: str | Path) -> list[str]:
    """
    Retrieves the immediate subfolders (first order) within a specified directory.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list[str]: A list of the full paths to the immediate subfolders.
    """
    subfolders: list[str] = list()
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            subfolders.append(entry_path)

    return subfolders


def gather_png_files_in_subfolder_dataframes(root_dir: str | Path, subfolder: str | Path) -> List[str]:
    """
    Search for all dataframe PNG images within the subfolder of a given directory.

    Recursively searches the specified root directory for a given subfolder
    and collects all the ".png" file names found within them.

    Args:
        root_dir: The path to the root directory to start the search.
        subfolder: The subfolder within root directory that contains the PNG images.

    Returns:
        A list of full paths to the PNG files found.

    Example:
        >>> result = gather_png_files_in_subfolder_dataframes("/path/to/folder")
        >>> for file_path in result:
        ...     print(file_path)
    """
    png_files: List[str] = list()

    # Walk through the directory structure starting from root_dir
    for current_root, dirs, files in os.walk(root_dir):
        # Identify the name of the current folder to check if it ends with "dataframes"
        folder_name: str = os.path.basename(current_root)
        if folder_name == subfolder:
            # Filter PNG files in the current folder
            for file_name in files:
                if file_name.lower().endswith('.png'):
                    # Store the absolute path to the PNG file
                    full_path: str = os.path.join(current_root, file_name)
                    png_files.append(full_path)
    return png_files


def generate_final_report(
    final_report_path: str | Path,
    query: str,
    summary_dict: Dict[str, Any],
    cache_dir: str | Path,
    yfinance_stocks_dir: str | Path,
) -> None:
    """
    Generate a final report in Markdown format.

    The report is generated by appending content and data from multiple sections, summaries, images, and tables.

    Args:
        final_report_path: The path to the final Markdown report file to write or append to.
        query: The main query or title for the report.
        summary_dict: A dictionary with:
            - Keys: file/report identifiers.
            - Values: objects containing at least:
            -- summary: a summary for the section.
            -- title: a title for the section.
        cache_dir: A directory where intermediate Markdown sections are cached.
        finance_stocks_dir: A directory containing data for Yahoo Finance stocks.
    """

    final_report_path = Path(final_report_path)
    cache_dir = Path(cache_dir)
    yfinance_stocks_dir = Path(yfinance_stocks_dir)

    # 1. Write the title of the final report and the high-level summaries.
    with final_report_path.open('a', encoding='utf-8') as f:
        f.write('# ' + query + '\n\n')
        for summary in summary_dict.values():
            f.write(summary.summary + '\n\n')

    # 2. Append each detailed section to the final report.
    for report, summary in summary_dict.items():
        # Create the section filename.
        section_filename = cache_dir / f'section_{Path(report).name}'

        # Read the section text from the cached file.
        with section_filename.open('r', encoding='utf-8') as f_section:
            section_content = f_section.read()

        # Append this section to the final Markdown report.
        with final_report_path.open('a', encoding='utf-8') as f_report:
            # Section title
            f_report.write('# ' + summary.title)
            # Source information based on the report name
            lower_report = report.lower()
            if 'generic' in lower_report:
                f_report.write('\n(Source: Google Search)')
            elif 'filing' in lower_report:
                f_report.write('\n(Source: SEC EDGAR)')
            elif 'yfinance_news' in lower_report:
                f_report.write('\n(Source: Yahoo Finance News)')
            elif 'yfinance_stocks' in lower_report:
                f_report.write('\n(Source: YFinance Stocks)')

            f_report.write('\n\n')
            # Section content
            f_report.write(section_content + '\n\n')

        # 3. If it's a YFinance stocks report, append images, tables, and an appendix.
        if 'report_yfinance_stocks' in report:
            # Collect the data frames (PNG images) from all subfolders for an appendix.
            company_tickers_list = list_first_order_subfolders(yfinance_stocks_dir)
            appendix_images_dict = dict()

            for image_path in company_tickers_list:
                # The dictionary key is the subfolder name, value is a list of PNG files in that subfolder.
                appendix_images_dict[Path(image_path).name] = gather_png_files_in_subfolder_dataframes(
                    yfinance_stocks_dir / Path(image_path).name, subfolder='dataframes'
                )

            # Read table data from the matching JSON file
            json_path = report.replace('txt', 'json')
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            table_dict = dict()
            table_markdown_dict = dict()
            try:
                # Ticker symbol is assumed to be part of the filename, split by underscores.
                ticker_symbol = report.split('_')[5]
            except IndexError:
                ticker_symbol = 'UNKNOWN'

            # Write a heading for the appended tables.
            with final_report_path.open('a', encoding='utf-8') as target:
                target.write(f'\n### Data Sources Tables - {ticker_symbol}\n\n')

            for table_name, table_str in data.items():
                # Extract the nested JSON strings and parse them.
                table_dict[table_name] = parse_table_str(table_str)
                # Convert parsed data to a Markdown table.
                table_markdown_dict[table_name] = dict_to_markdown_table(table_dict[table_name], table_name)
                with final_report_path.open('a', encoding='utf-8') as target:
                    target.write(table_markdown_dict[table_name])
                    target.write('\n\n')

            # If any images were found, write them to the final report as an appendix.
            if len(appendix_images_dict) > 0:
                for company_ticker, images_list in appendix_images_dict.items():
                    with final_report_path.open('a', encoding='utf-8') as target:
                        target.write(f'\n### Data Sources Images - {company_ticker}\n\n')
                    convert_file_of_image_paths_to_markdown(images_list, final_report_path, f'Image {company_ticker}')


def delete_temp_dir(temp_dir: str, verbose: bool = False) -> None:
    """Delete the temporary directory and its contents."""

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            if verbose:
                logger.info(f'Temporary directory {temp_dir} deleted.')
        except:
            if verbose:
                logger.warning(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """Schedule the deletion of the temporary directory after a delay."""

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir=temp_dir, verbose=False).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()


def create_yfinance_stock_dir(cache_dir: Path) -> Path:
    """Create the directory for Yahoo Finance stock images."""

    return cache_dir / 'yfinance_stocks'


def create_log_path(cache_dir: Path) -> str:
    """Create the CrewAI log file JSON path."""

    # CrewAI logging JSON file
    return str(cache_dir / 'output_log_file.json')
