
import os
from dotenv import load_dotenv, find_dotenv
import yaml


def load_env():
    _ = load_dotenv(find_dotenv())



def get_sambanova_api_key():
    load_env()
    sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
    return sambanova_api_key


def get_serper_api_key():
    load_env()
    serper_api_key = os.getenv("SERPER_API_KEY")
    return serper_api_key


# break line every 80 characters if line is longer than 80 characters
# don't break in the middle of a word
def pretty_print_result(result):
  parsed_result = []
  for line in result.split('\n'):
      if len(line) > 80:
          words = line.split(' ')
          new_line = ''
          for word in words:
              if len(new_line) + len(word) + 1 > 80:
                  parsed_result.append(new_line)
                  new_line = word
              else:
                  if new_line == '':
                      new_line = word
                  else:
                      new_line += ' ' + word
          parsed_result.append(new_line)
      else:
          parsed_result.append(line)
  return "\n".join(parsed_result)


def update_task_output_format(tasks_config: dict, task_name: str, format_expected: str) -> dict:
    """
    Updates the expected_output of a specific task to include structured output formatting.
    
    Args:
        tasks_config (dict): The tasks configuration dictionary (already loaded from YAML)
        task_name (str): Name of the task to update
        format_expected (str): The expected output format (could be Pydantic, JSON schema, or any other format)
    
    Returns:
        dict: Updated tasks configuration
    """
    if task_name not in tasks_config:
        raise KeyError(f"Task '{task_name}' not found in configuration")
    
    # Get the original expected output
    original_output = tasks_config[task_name]['expected_output']
    
    # Add the structured output requirement
    structured_output = (
        f"{original_output}\n\n"
        f"You MUST structure your output as follows:\n\n"
        f"{escape_format_instructions(format_expected)}"
    )
    
    # Update the configuration
    tasks_config[task_name]['expected_output'] = structured_output
    
    return tasks_config

def escape_format_instructions(format_instructions: str) -> str:
    """
    Escapes curly braces in the format instructions to prevent CrewAI from interpreting them as variables
    """
    # Replace { with {{ and } with }} to escape them
    escaped = format_instructions.replace('{', '{{').replace('}', '}}')
    
    # Example needs to be escaped too
    example = """{{
        "tasks": [
            {{
                "task_name": "Example Task",
                "estimated_time_hours": 2.5,
                "required_resources": ["Developer", "Designer"]
            }}
        ],
        "milestones": [
            {{
                "milestone_name": "Example Milestone",
                "tasks": ["Example Task"]
            }}
        ]
    }}"""
    
    return f"""
Please provide your output in the following format:
{escaped}

Example output:
{example}
"""