# tokenizer.py
import os
from transformers import AutoTokenizer

os.environ['TRANSFORMERS_CACHE'] = './cache/'

class SingletonTokenizer:
    _instance = None

    def __init__(self, path_to_tokenizer):
        if SingletonTokenizer._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LLAMA_2_70B_CHAT_PATH = path_to_tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_70B_CHAT_PATH)
            SingletonTokenizer._instance = self

    @staticmethod
    def get_instance():
        if SingletonTokenizer._instance is None:
            SingletonTokenizer()
        return SingletonTokenizer._instance.tokenizer
