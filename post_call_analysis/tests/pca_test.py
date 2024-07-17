#!/usr/bin/env python3
"""
Post Call Analysis (PCA) Test Script

This script tests the functionality of the Post Call Analysis using unittest.
It loads transcripts and tests the analysis capabilities.

Usage:
    python tests/pca_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""


import os
import sys
import shutil
import time
import unittest
import logging
from typing import List, Dict, Any

current_dir = os.getcwd()
print(current_dir)
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import glob
import pandas as pd

from post_call_analysis.src import analysis, plot, asr

audio_save_location=(os.path.join(kit_dir,"data/conversations/audio"))
transcript_save_location=(os.path.join(kit_dir,"data/conversations/transcription"))

def convert_to_dialogue_structure(transcription):
    dialogue = ''  
    for _, row in transcription.iterrows():
        speaker = str(row['speaker'])
        text = str(row['text'])
        dialogue += speaker + ': ' + text + '\n'   
    return dialogue

transcription_path = os.path.join(transcript_save_location,'911_call.csv')
transcription=pd.read_csv(transcription_path)

dialogue = convert_to_dialogue_structure(transcription)
print('Petro: finish converting dialogue')
conversation = analysis.load_conversation(dialogue, transcription_path)
print('Petro: finish loading conversation')
conversation_chunks = analysis.get_chunks(conversation)
print('Petro: finish chunking')

#result=analysis.call_analysis_parallel(conversation_chunks, documents_path=facts_path, facts_urls=facts_urls, procedures_path=procedures_path, classes_list=classes, entities_list=entities, sentiment_list=sentiments)
#print('Petro: finish analysis')