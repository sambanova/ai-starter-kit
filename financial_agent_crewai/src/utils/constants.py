from pathlib import Path

# Specify the directory and file path
CACHE_DIR = Path('financial_agent_crewai/cache/')

# User query
# USER_QUERY = 'What are the differences between the 2023 and 2024 10-K filings for Meta?'
USER_QUERY = 'What are the differences between Apple and Google filings in 2024?'

# Comparison query
COMPARISON_QUERY = (
    'Please write a conclusion/summary for the provided context, '
    f'specifically focusing around the user query: {USER_QUERY}'
)
