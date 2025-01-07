"""
Configuration module for Language Model (LLM) settings.

This module initializes and configures the Language Model instance used throughout
the educational content generation system. It uses configuration settings from
the main config module.
"""

from crewai import LLM

from .config import LLM_CONFIG

# Initialize the Language Model with configuration settings
llm = LLM(model=LLM_CONFIG['model'], api_key=LLM_CONFIG['api_key'])
