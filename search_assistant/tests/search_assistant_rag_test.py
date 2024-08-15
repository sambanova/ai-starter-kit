#!/usr/bin/env python3
"""
Search Assistant (RAG Query) Test Script

This script tests the functionality of the Search Assistant kit (search and scrape sites) using unittest.

Usage:
    python tests/search_assistant_rag_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import os
import sys
import unittest
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
file_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(file_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from search_assistant.src.search_assistant import SearchAssistant

method = 'rag_query'
input_disabled = False
tool = ['serpapi'] # serpapi, serper, openserp
search_engine = 'google' # google, bing, baidu
max_results = 5
query = 'Albert Einstein'

search_assistant = SearchAssistant()

scraper_state = search_assistant.search_and_scrape(
                            query=query,
                            search_method=tool[0],
                            max_results=max_results,
                            search_engine=search_engine,
                        )

user_question = 'who is Albert Einsten?'
response = search_assistant.retrieval_call(user_question)
print(response["source_documents"]) # list[Document]
print(response["answer"]) # str

