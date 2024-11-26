#!/usr/bin/env python
import sys
import warnings

from financial_agent.src.crew import LatestAiDevelopment

warnings.filterwarnings('ignore', category=SyntaxWarning, module='pysbd')

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run() -> None:
    """
    Run the crew.
    """
    inputs = {'user_query': 'What is the research and development strategy at Meta?'}
    LatestAiDevelopment().crew().kickoff(inputs=inputs)


def train() -> None:
    """
    Train the crew for a given number of iterations.
    """
    inputs = {'topic': 'AI LLMs'}
    try:
        LatestAiDevelopment().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f'An error occurred while training the crew: {e}')


def replay() -> None:
    """
    Replay the crew execution from a specific task.
    """
    try:
        LatestAiDevelopment().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f'An error occurred while replaying the crew: {e}')


def test() -> None:
    """
    Test the crew execution and returns the results.
    """
    inputs = {'topic': 'AI LLMs'}
    try:
        LatestAiDevelopment().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f'An error occurred while replaying the crew: {e}')


if __name__ == '__main__':
    run()
