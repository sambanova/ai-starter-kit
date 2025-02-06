import base64
import datetime
import json
import os
import re
import shutil
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Match, Optional, Union

import markdown
import weasyprint  # type: ignore
from bs4 import BeautifulSoup


def clear_directory(directory: Union[str, Path]) -> Optional[str]:
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
    """Convert a millisecond-based Unix timestamp into a human-readable YYYY-MM-DD date string in UTC."""

    dt = datetime.datetime.utcfromtimestamp(ts_millis / 1000.0)
    return dt.strftime('%Y-%m-%d')


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
    columns = table_data.get('columns', [])
    index = table_data.get('index', [])
    data = table_data.get('data', [])

    # Start building Markdown
    lines: List[str] = []

    # Title as a second-level heading
    lines.append(f'## {title}')

    # Header row
    header = ['Date'] + columns
    lines.append('| ' + ' | '.join(header) + ' |')
    # Markdown requires a separator row of dashes
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    # Populate each row
    for row_idx, row_data in enumerate(data):
        # Convert the timestamp in 'index' to a date string
        date_str = timestamp_to_date_string(index[row_idx])
        # Convert each value in row_data to a string
        row_cells = [str(val) for val in row_data]
        lines.append('| ' + date_str + ' | ' + ' | '.join(row_cells) + ' |')

    # Combine all lines into a single string
    return '\n'.join(lines)


def convert_image_path_to_markdown(image_path: str, alt_text: str = 'Image') -> str:
    """
    Convert an image path to a Markdown image reference.

    Args:
        image_path: The filesystem path (or URL) pointing to the image.
        alt_text: The alt text describing the image. Defaults to "Image".

    Returns:
        str: A string in the Markdown format for embedding images.

    Example:
        >>> convert_image_path_to_markdown("plots/graph.png", "Stock graph")
        '![Stock graph](plots/graph.png)'
    """
    # In Markdown: ![alt_text](image_path)
    return f'![{alt_text}]({image_path})'


def convert_file_of_image_paths_to_markdown(input_file: str, output_file: str, alt_text: str = 'Image') -> None:
    """
    Reads an input text file, searches for any .png paths embedded in the text,
    and writes the resulting text to an output file. In the resulting text,
    each found .png path is replaced with a Markdown image reference of the form:
        ![<alt text>](<image path>)

    Args:
        input_file: Path to the input text file.
        output_file: Path to the output file where Markdown statements will be written.
        alt_text: The alt text to use for each image (defaults to "Image").

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
    # Read entire file contents
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Regex pattern for any sequence of non-whitespace characters ending in .png
    pattern = r'(\S+\.png)'

    # Replace each .png path with the Markdown image reference
    transformed_text = re.sub(pattern, lambda match: f'![{alt_text}]({match.group(1)})', text)

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
    for table in soup.find_all('table'):
        # Add styling classes or IDs as needed
        table['class'] = (table.get('class') or []) + ['table', 'table-striped', 'table-bordered']

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
        size: A4;        /* Adjust if you'd like a different format */
        margin: 2cm;     /* Adjust margins as desired */
    }
    img {
        max-width: 100%; /* Scale down images that exceed page width */
        height: auto;    /* Preserve aspect ratio */
        display: block;  /* Avoid text wrapping around large images */
    }
    """

    # Create a WeasyPrint CSS object from the style string
    stylesheet = weasyprint.CSS(string=style)

    # Build the PDF in-memory, applying the custom stylesheet
    pdf_data = weasyprint.HTML(string=html_str).write_pdf(target=output_file, stylesheets=[stylesheet])

    return pdf_data


@contextmanager
def st_capture(output_func: Any) -> Generator[StringIO, None, None]:
    """
    Context manager for capturing stdout and redirecting to Streamlit.

    Args:
        output_func (Callable[[str], None]): Function to handle captured output.

    Yields:
        StringIO: String buffer containing captured output.
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            # Each time something is written to stdout,
            # we send it to Streamlit via `output_func`.
            output_func(stdout.getvalue() + '\n#####\n')
            return ret

        stdout.write = new_write  # type: ignore
        yield stdout
