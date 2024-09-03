from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid


class LlaVaData:

    def download_hf_data(self, dataset_name, output_dir):