import os
import sys
import yaml
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

class PDFRetrieval():
    def __init__(self):
        pass

    def _get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        retreival_info = config["retrieval"]
        loader = config["loader"]
        
        return api_info, llm_info, retreival_info, loader
    
    #WIP
    
    
    