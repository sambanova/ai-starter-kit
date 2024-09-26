import yaml
import glob
from typing import Dict
import pdfkit
from sec_edgar_downloader import Downloader


class SECTools:

    def __init__(self,
                config_path: str) -> None:
        
        self.configs = self.load_config(config_path)


    @staticmethod
    def load_config(filename: str) -> dict:
        """
        Loads a YAML configuration file and returns its contents as a dictionary.

        Args:
            filename: The path to the YAML configuration file.

        Returns:
            A dictionary containing the configuration file's contents.
        """

        try:
            with open(filename, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The YAML configuration file {filename} was not found.')
        except yaml.YAMLError as e:
            raise RuntimeError(f'Error parsing YAML file: {e}')
        
    def download_filings(self, download_folder: str) -> None:

        dl = Downloader(company_name=self.configs["sec"]["company"],
                        email_address=self.configs["sec"]["email"],
                        download_folder=download_folder)
        
        for ticker in self.configs["sec"]["tickers"]:
            for form_type in self.configs["sec"]["form_types"]:

                dl.get(
                    form=form_type,
                    ticker_or_cik=ticker,
                    after=self.configs["sec"]["start_date"],
                    before=self.configs["sec"]["end_date"]
                )

    def _read_txt_file(self, filename: str) -> str:

        with open(filename, 'r') as file:

            return file.read()
    
    #TODO: Better handling of xbrl content.
    def convert_txt_to_pdf(self,
                           data_directory: str,
                           options: Dict[str, str] = {
                                        'page-size': 'Letter',
                                        'margin-top': '0.75in',
                                        'margin-right': '0.75in',
                                        'margin-bottom': '0.75in',
                                        'margin-left': '0.75in',
                                        'encoding': "UTF-8",
                                        'no-outline': None
                                    }) -> True:

        files = glob.glob(data_directory + "**/**",
                                  recursive=True)
        files = [file for file in files if file.endswith(".txt")]

        for filename in files:
            print(f"Converting {filename} to pdf in the same location")
            text = self._read_txt_file(filename)
            try:
                pdfkit.from_string(text, 
                                filename.split(".txt")[0] + '.pdf', 
                                options=options)
            except OSError as e:
                print(f"Error: {e} occurred.  Check outputs.")



    

    
    
